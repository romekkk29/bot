#!/usr/bin/env python3
"""CLI de prueba: Groq + tool calling contra stubs (luego tus APIs)."""

from __future__ import annotations

import os
import sys
from datetime import date

from dotenv import load_dotenv
from groq import Groq

from tools import TOOLS, data_backend_label, dispatch_tool

def _build_system_prompt() -> str:
    hoy = date.today().isoformat()
    return f"""Sos el asistente del ERP de la empresa.
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


def run_turn(client: Groq, model: str, messages: list[dict]) -> str:
    while True:
        response = client.chat.completions.create(
            model=model,
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


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Definí GROQ_API_KEY (copiá .env.example a .env).", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)

    print(
        f"ERP bot (Groq). Datos: {data_backend_label()}.\n"
        "Escribí tu pregunta; vacío o 'salir' para terminar.\n"
    )

    history: list[dict] = [{"role": "system", "content": _build_system_prompt()}]

    while True:
        try:
            line = input("vos> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("salir", "exit", "quit"):
            break

        history.append({"role": "user", "content": line})
        try:
            reply = run_turn(client, model, list(history))
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
