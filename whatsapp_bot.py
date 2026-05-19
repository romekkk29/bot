#!/usr/bin/env python3
"""Bot de WhatsApp integrado con ERP usando FastAPI."""

from __future__ import annotations

import os
import sys
from datetime import date
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel

from tools import TOOLS, data_backend_label, dispatch_tool

# Cargar variables de entorno
load_dotenv()

# Configuración
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
WHATSAPP_API_KEY = os.environ.get("WHATSAPP_API_KEY")
WHATSAPP_API_URL = os.environ.get("WHATSAPP_API_URL", "https://api.kapso.io")

if not GROQ_API_KEY:
    print("ERROR: Definí GROQ_API_KEY en .env", file=sys.stderr)
    sys.exit(1)

# Inicializar cliente Groq
client = Groq(api_key=GROQ_API_KEY)

# Inicializar FastAPI
app = FastAPI(title="ERP WhatsApp Bot")


# Modelos de datos
class WhatsAppMessage(BaseModel):
    from_number: str
    message: str
    message_id: str | None = None


class WhatsAppWebhook(BaseModel):
    event: str
    data: dict[str, Any]


# Historial de conversaciones por número de teléfono
conversation_history: dict[str, list[dict]] = {}


def _build_system_prompt() -> str:
    hoy = date.today().isoformat()
    return f"""Sos el asistente del ERP de la empresa por WhatsApp.
Respondés en español, claro y breve.
Fecha actual de referencia: {hoy}.
Para datos de ventas o compras (totales, listados, conteo por estado, líneas de una OC), stock disponible, productos bajo mínimo, movimientos de cardex, productos, proveedores o clientes NO inventes números: usá las herramientas disponibles.
Si el usuario pide períodos relativos (ej: 'último mes'), inferí fechas usando la fecha actual y llamá la tool.
Si usaste herramientas y devolvieron datos, tratá esos datos como reales de esta consulta.
No agregues frases de descargo genéricas como 'es solo un ejemplo', 'puede variar' o similares.
Si falta información crítica que no puedas inferir, recién ahí pedila al usuario."""


def _assistant_message_to_dict(msg) -> dict:
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
    return d


def run_turn(messages: list[dict]) -> str:
    while True:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            messages.append(_assistant_message_to_dict(msg))
            for tc in msg.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                output = dispatch_tool(name, raw_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )
            continue

        return (msg.content or "").strip()


async def send_whatsapp_message(to_number: str, message: str) -> bool:
    """Envía mensaje a través de la API de WhatsApp/Kapso."""
    if not WHATSAPP_API_KEY:
        print("WARNING: WHATSAPP_API_KEY no configurado, mensaje no enviado", file=sys.stderr)
        return False
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{WHATSAPP_API_URL}/messages",
                headers={
                    "Authorization": f"Bearer {WHATSAPP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "to": to_number,
                    "message": message,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"ERROR enviando mensaje a WhatsApp: {e}", file=sys.stderr)
        return False


@app.get("/")
async def root():
    """Endpoint de health check."""
    return {
        "status": "ok",
        "backend": data_backend_label(),
        "model": GROQ_MODEL,
    }


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(webhook: WhatsAppWebhook):
    """Recibe webhooks de WhatsApp/Kapso."""
    print(f"[WEBHOOK] Recibido: {webhook}", file=sys.stderr)
    try:
        # Extraer datos del webhook (formato puede variar según proveedor)
        if webhook.event == "message_received":
            data = webhook.data
            from_number = data.get("from", "").replace("whatsapp:", "")
            message = data.get("message", "")
            message_id = data.get("id")
            
            print(f"[WEBHOOK] De: {from_number}, Mensaje: {message}", file=sys.stderr)
            
            if not from_number or not message:
                print(f"[WEBHOOK] Error: Missing from_number or message", file=sys.stderr)
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing from_number or message"}
                )
            
            # Obtener o crear historial
            if from_number not in conversation_history:
                conversation_history[from_number] = [
                    {"role": "system", "content": _build_system_prompt()}
                ]
            
            # Agregar mensaje del usuario
            conversation_history[from_number].append({"role": "user", "content": message})
            
            # Procesar con el bot
            try:
                print(f"[WEBHOOK] Procesando mensaje...", file=sys.stderr)
                reply = run_turn(list(conversation_history[from_number]))
                conversation_history[from_number].append({"role": "assistant", "content": reply})
                
                print(f"[WEBHOOK] Respuesta: {reply}", file=sys.stderr)
                
                # Enviar respuesta a WhatsApp
                sent = await send_whatsapp_message(from_number, reply)
                print(f"[WEBHOOK] Enviado a WhatsApp: {sent}", file=sys.stderr)
                
                return {"status": "success", "reply": reply}
            except Exception as e:
                print(f"[WEBHOOK] ERROR procesando mensaje: {e}", file=sys.stderr)
                conversation_history[from_number].pop()  # Remover mensaje del usuario
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error processing message: {str(e)}"}
                )
        
        print(f"[WEBHOOK] Evento ignorado: {webhook.event}", file=sys.stderr)
        return {"status": "ignored", "event": webhook.event}
    
    except Exception as e:
        print(f"[WEBHOOK] ERROR en webhook: {e}", file=sys.stderr)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/message")
async def direct_message(msg: WhatsAppMessage):
    """Endpoint directo para enviar mensajes al bot (para testing)."""
    try:
        # Obtener o crear historial
        if msg.from_number not in conversation_history:
            conversation_history[msg.from_number] = [
                {"role": "system", "content": _build_system_prompt()}
            ]
        
        # Agregar mensaje del usuario
        conversation_history[msg.from_number].append({"role": "user", "content": msg.message})
        
        # Procesar con el bot
        reply = run_turn(list(conversation_history[msg.from_number]))
        conversation_history[msg.from_number].append({"role": "assistant", "content": reply})
        
        return {"status": "success", "reply": reply}
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/history/{phone_number}")
async def get_history(phone_number: str):
    """Obtiene historial de conversación de un número."""
    if phone_number not in conversation_history:
        return {"history": []}
    
    # Retornar sin el system prompt
    history = conversation_history[phone_number][1:]
    return {"history": history}


@app.delete("/history/{phone_number}")
async def clear_history(phone_number: str):
    """Limpia el historial de un número."""
    if phone_number in conversation_history:
        del conversation_history[phone_number]
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    
    print(f"Iniciando ERP WhatsApp Bot (Groq: {GROQ_MODEL})")
    print(f"Backend de datos: {data_backend_label()}")
    print(f"Servidor en http://0.0.0.0:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
