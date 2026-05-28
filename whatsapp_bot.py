#!/usr/bin/env python3
"""Bot de WhatsApp integrado con ERP usando FastAPI."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel, Field

from tools import TOOLS, data_backend_label, dispatch_tool

# Cargar variables de entorno
load_dotenv()

# Configuración
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
WHATSAPP_API_KEY = os.environ.get("WHATSAPP_API_KEY")
WHATSAPP_API_URL = os.environ.get("WHATSAPP_API_URL", "https://api.kapso.ai/meta/whatsapp/v24.0")

if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        print("ERROR: Definí GEMINI_API_KEY en .env", file=sys.stderr)
        sys.exit(1)
    client: Any = None
    print(f"[LLM] Usando Gemini ({GEMINI_MODEL})", file=sys.stderr)
else:
    if not GROQ_API_KEY:
        print("ERROR: Definí GROQ_API_KEY en .env", file=sys.stderr)
        sys.exit(1)
    client = Groq(api_key=GROQ_API_KEY)
    print(f"[LLM] Usando Groq ({GROQ_MODEL})", file=sys.stderr)

# Inicializar FastAPI
app = FastAPI(title="ERP WhatsApp Bot")


# Modelos de datos
class WhatsAppMessage(BaseModel):
    from_number: str
    message: str
    message_id: str | None = None


class KapsoMessageText(BaseModel):
    body: str | None = None


class KapsoMessage(BaseModel):
    id: str | None = None
    from_: str | None = Field(default=None, alias="from")
    text: KapsoMessageText | None = None
    type: str | None = None

    model_config = {"populate_by_name": True}


class KapsoWebhook(BaseModel):
    message: KapsoMessage | None = None
    phone_number_id: str | None = None
    is_new_conversation: bool | None = None
    model_config = {"extra": "allow"}


# Historial de conversaciones por número de teléfono
conversation_history: dict[str, list[dict]] = {}

# Deduplicación de webhooks ya procesados
processed_ids: set[str] = set()

# Máximo de mensajes (sin contar system prompt) a mantener por conversación
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "6"))

# Máximo de caracteres por resultado de tool (evita payloads enormes de Supabase)
MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "3000"))


def _build_system_prompt() -> str:
    hoy = date.today().isoformat()
    return f"""Sos el asistente del ERP de la empresa por WhatsApp.
Respondés en español, claro y breve.
Fecha actual de referencia: {hoy}.
Para datos de ventas o compras (totales, listados, conteo por estado, líneas de una OC), stock disponible, productos bajo mínimo, movimientos de cardex, productos, proveedores o clientes NO inventes números: usá las herramientas disponibles.
Si el usuario pide períodos relativos (ej: 'último mes'), inferí fechas usando la fecha actual y llamá la tool.
Si usaste herramientas y devolvieron datos, tratá esos datos como reales de esta consulta.
No agregues frases de descargo genéricas como 'es solo un ejemplo', 'puede variar' o similares.
Si falta información crítica que no puedas inferir, recién ahí pedila al usuario.

REGLA CRÍTICA — consultas sin filtro:
Si el usuario pide "todas las ventas", "todo el detalle", "todos los productos", "todos los clientes" u otra consulta masiva SIN un filtro concreto (fecha, nombre, período, estado), NO ejecutes la herramienta.
En cambio, respondé explicando qué filtros podés aplicar y pedí al menos uno. Ejemplos de filtros válidos: rango de fechas, nombre de producto/proveedor/cliente, estado de orden, o un período como 'hoy', 'esta semana', 'este mes'.
Sí podés hacer consultas de resumen/totales o conteos sin filtro de fecha, ya que no devuelven filas individuales."""


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


def _schema_to_genai(schema: dict) -> Any:
    """Convert JSON Schema dict to google.genai types.Schema."""
    from google.genai import types as _gt
    TYPE_MAP = {
        "string": _gt.Type.STRING, "integer": _gt.Type.INTEGER,
        "number": _gt.Type.NUMBER, "boolean": _gt.Type.BOOLEAN,
        "array": _gt.Type.ARRAY, "object": _gt.Type.OBJECT,
    }
    src = dict(schema)
    if "anyOf" in src and "type" not in src:
        for opt in src["anyOf"]:
            if isinstance(opt, dict) and opt.get("type") not in (None, "null"):
                src = {k: v for k, v in src.items() if k != "anyOf"}
                src.update(opt)
                break
    t = str(src.get("type", "object")).lower()
    kwargs: dict[str, Any] = {"type": TYPE_MAP.get(t, _gt.Type.STRING)}
    if d := src.get("description"):
        kwargs["description"] = d
    if props := src.get("properties"):
        kwargs["properties"] = {k: _schema_to_genai(v) for k, v in props.items()}
    if req := src.get("required"):
        kwargs["required"] = list(req)
    if items := src.get("items"):
        kwargs["items"] = _schema_to_genai(items)
    return _gt.Schema(**kwargs)


def _build_genai_tool() -> Any:
    from google.genai import types as _gt
    return _gt.Tool(
        function_declarations=[
            _gt.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"]["description"],
                parameters=_schema_to_genai(t["function"].get("parameters", {})),
            )
            for t in TOOLS
        ]
    )


def _run_turn_gemini(messages: list[dict]) -> str:
    """Run one conversational turn using the google-genai SDK."""
    from google import genai as _gg
    from google.genai import types as _gt

    system_parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    system_instruction = "\n".join(system_parts) or None

    contents = [
        _gt.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[_gt.Part.from_text(text=m.get("content") or "")],
        )
        for m in messages if m["role"] != "system"
    ]

    gemini = _gg.Client(api_key=GEMINI_API_KEY)
    config = _gt.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[_build_genai_tool()],
        temperature=0.2,
    )

    while True:
        response = gemini.models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=config
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts
        fn_calls = [p.function_call for p in parts if p.function_call and p.function_call.name]

        if not fn_calls:
            return "".join(p.text for p in parts if p.text).strip()

        contents.append(candidate.content)
        fn_parts = []
        for fc in fn_calls:
            args_json = json.dumps(dict(fc.args))
            output = dispatch_tool(fc.name, args_json)
            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + "...[truncado]"
            print(f"[GEMINI] tool: {fc.name}({args_json[:80]})", file=sys.stderr)
            fn_parts.append(_gt.Part.from_function_response(
                name=fc.name, response={"result": output}
            ))
        contents.append(_gt.Content(role="user", parts=fn_parts))


def run_turn(messages: list[dict]) -> str:
    if LLM_PROVIDER == "gemini":
        return _run_turn_gemini(messages)
    # Groq (OpenAI-compatible)
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
                if len(output) > MAX_TOOL_OUTPUT_CHARS:
                    output = output[:MAX_TOOL_OUTPUT_CHARS] + "...[truncado]"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )
            continue

        return (msg.content or "").strip()


async def send_whatsapp_message(to_number: str, message: str, phone_number_id: str | None = None) -> bool:
    """Envía mensaje a través de la API de Kapso."""
    if not WHATSAPP_API_KEY:
        print("WARNING: WHATSAPP_API_KEY no configurado, mensaje no enviado", file=sys.stderr)
        return False

    try:
        import httpx

        if not phone_number_id:
            print("[SEND] ERROR: phone_number_id no disponible", file=sys.stderr)
            return False

        url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"

        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                url,
                headers={
                    "X-API-Key": WHATSAPP_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to_number,
                    "type": "text",
                    "text": {"body": message},
                },
                timeout=30.0,
            )
            print(f"[SEND] Status: {response.status_code}, Body: {response.text}", file=sys.stderr)
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"[SEND] ERROR: {e}", file=sys.stderr)
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
async def whatsapp_webhook(request: Request):
    """Recibe webhooks de WhatsApp/Kapso."""
    try:
        body = await request.json()

        # Ignorar webhooks de prueba de Kapso
        if body.get("test"):
            print("[WEBHOOK] Ignorado: webhook de test", file=sys.stderr)
            return {"status": "ignored", "reason": "test"}

        msg = body.get("message", {}) or {}
        msg_id = msg.get("id", "")

        # Deduplicación: ignorar webhooks ya procesados
        if msg_id and msg_id in processed_ids:
            print(f"[WEBHOOK] Duplicado ignorado: {msg_id}", file=sys.stderr)
            return {"status": "ignored", "reason": "duplicate"}
        if msg_id:
            processed_ids.add(msg_id)
            if len(processed_ids) > 10000:
                processed_ids.clear()

        # Filtrar mensajes salientes (outbound) para evitar ecos
        kapso_meta = msg.get("kapso") or {}
        if kapso_meta.get("direction") == "outbound":
            print("[WEBHOOK] Ignorado: mensaje outbound", file=sys.stderr)
            return {"status": "ignored", "reason": "outbound"}

        from_number = str(msg.get("from", "")).strip()
        phone_number_id = str(body.get("phone_number_id") or "").strip() or None
        text_obj = msg.get("text") or {}
        message_text = str(text_obj.get("body", "")).strip()

        print(f"[WEBHOOK] De: {from_number!r}, Texto: {message_text!r}", file=sys.stderr)

        if not from_number or not message_text:
            print("[WEBHOOK] Ignorado: sin número o texto", file=sys.stderr)
            return {"status": "ignored"}

        # Solo procesar mensajes de texto entrantes
        if msg.get("type") != "text":
            print(f"[WEBHOOK] Ignorado: tipo {msg.get('type')}", file=sys.stderr)
            return {"status": "ignored", "reason": "not text"}

        # Obtener o crear historial
        if from_number not in conversation_history:
            conversation_history[from_number] = [
                {"role": "system", "content": _build_system_prompt()}
            ]

        conversation_history[from_number].append({"role": "user", "content": message_text})

        # Recortar historial: conservar system prompt + últimos MAX_HISTORY_TURNS mensajes
        hist = conversation_history[from_number]
        system_msgs = [m for m in hist if m["role"] == "system"]
        non_system = [m for m in hist if m["role"] != "system"]
        if len(non_system) > MAX_HISTORY_TURNS:
            non_system = non_system[-MAX_HISTORY_TURNS:]
        conversation_history[from_number] = system_msgs + non_system

        try:
            print("[WEBHOOK] Procesando con LLM...", file=sys.stderr)
            reply = run_turn(list(conversation_history[from_number]))
            conversation_history[from_number].append({"role": "assistant", "content": reply})

            print(f"[WEBHOOK] Respuesta: {reply}", file=sys.stderr)

            sent = await send_whatsapp_message(from_number, reply, phone_number_id)
            print(f"[WEBHOOK] Enviado a WhatsApp: {sent}", file=sys.stderr)

            return {"status": "success", "reply": reply}
        except Exception as e:
            err_str = str(e)
            print(f"[WEBHOOK] ERROR LLM: {err_str}", file=sys.stderr)
            conversation_history[from_number].pop()
            # Notificar al usuario si se agotó el límite de tokens
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                aviso = (
                    "⚠️ El asistente está temporalmente no disponible por límite de uso. "
                    "Por favor intentá de nuevo en unos minutos."
                )
                await send_whatsapp_message(from_number, aviso, phone_number_id)
            # Siempre retornar 200 para evitar reintentos de Kapso
            return {"status": "error", "detail": err_str}

    except Exception as e:
        print(f"[WEBHOOK] ERROR general: {e}", file=sys.stderr)
        # Siempre 200 para evitar reintentos
        return {"status": "error", "detail": str(e)}


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
