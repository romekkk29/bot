#!/usr/bin/env python3
"""CLI de prueba: Groq + tool calling contra stubs (luego tus APIs)."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from tools import TOOLS, data_backend_label, dispatch_tool

MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "2000"))


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


def _schema_to_genai(schema: dict) -> Any:
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


def run_turn_gemini(model_name: str, api_key: str, messages: list[dict]) -> str:
    from google import genai as _gg
    from google.genai import types as _gt

    system_parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    system_instruction = "\n".join(system_parts) or None
    tool = _gt.Tool(
        function_declarations=[
            _gt.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"]["description"],
                parameters=_schema_to_genai(t["function"].get("parameters", {})),
            )
            for t in TOOLS
        ]
    )
    contents = [
        _gt.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[_gt.Part.from_text(text=m.get("content") or "")],
        )
        for m in messages if m["role"] != "system"
    ]
    gemini = _gg.Client(api_key=api_key)
    config = _gt.GenerateContentConfig(
        system_instruction=system_instruction, tools=[tool], temperature=0.2
    )
    while True:
        # ── DEBUG: prompt completo ──────────────────────────────────────────
        total_chars = 0
        print("\n" + "═" * 60, flush=True)
        if system_instruction:
            print(f"[SYSTEM] ({len(system_instruction)}c)\n{system_instruction}", flush=True)
            total_chars += len(system_instruction)
        for i, c in enumerate(contents):
            role = getattr(c, "role", "?")
            for p in (c.parts or []):
                if hasattr(p, "text") and p.text:
                    txt = p.text
                    print(f"[{role.upper()} #{i}] ({len(txt)}c)\n{txt}", flush=True)
                    total_chars += len(txt)
                elif hasattr(p, "function_call") and p.function_call:
                    fc = p.function_call
                    print(f"[{role.upper()} #{i}] tool_call: {fc.name}({dict(fc.args)})", flush=True)
                elif hasattr(p, "function_response") and p.function_response:
                    fr = p.function_response
                    resp_str = str(fr.response)
                    total_chars += len(resp_str)
                    print(f"[{role.upper()} #{i}] tool_result: {fr.name} → {resp_str[:200]}", flush=True)
        print(f"── TOTAL: {len(contents)} contenidos, ~{total_chars}c (~{total_chars//4} tokens est.) ──", flush=True)
        print("═" * 60 + "\n", flush=True)
        # ── FIN DEBUG ───────────────────────────────────────────────────────
        response = gemini.models.generate_content(
            model=model_name, contents=contents, config=config
        )
        # ── TOKEN USAGE REAL ────────────────────────────────────────────────
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            u = response.usage_metadata
            print(
                f"[TOKENS] input={u.prompt_token_count}  output={u.candidates_token_count}  "
                f"total={u.total_token_count}  (de input, tools schemas≈{u.prompt_token_count - total_chars//4} tokens est.)",
                flush=True,
            )
        # ────────────────────────────────────────────────────────────────────
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
            print(f"  → tool: {fc.name}({args_json[:80]})")
            print(f"     ← {output[:300]}")
            fn_parts.append(_gt.Part.from_function_response(
                name=fc.name, response={"result": output}
            ))
        contents.append(_gt.Content(role="user", parts=fn_parts))


def run_turn_ollama(base_url: str, model: str, messages: list[dict]) -> str:
    from openai import OpenAI as _OAI

    ollama_client = _OAI(base_url=base_url, api_key="ollama")
    if "qwen3" in model.lower():
        msgs = [
            {**m, "content": "/no_think\n" + (m.get("content") or "")}
            if m["role"] == "system" else m
            for m in messages
        ]
    else:
        msgs = messages

    while True:
        response = ollama_client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            msgs.append(_assistant_message_to_dict(msg))
            for tc in msg.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                output = dispatch_tool(name, raw_args)
                if len(output) > MAX_TOOL_OUTPUT_CHARS:
                    output = output[:MAX_TOOL_OUTPUT_CHARS] + "...[truncado]"
                print(f"  → tool: {name}({raw_args[:80]})")
                print(f"     ← {output[:300]}")
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            continue

        return (msg.content or "").strip()


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

    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Definí GEMINI_API_KEY en .env.", file=sys.stderr)
            sys.exit(1)
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        client: Groq | None = None
        label = f"Gemini ({model})"
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        api_key = None
        client = None
        label = f"Ollama local ({model})"
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Definí GROQ_API_KEY (copiá .env.example a .env).", file=sys.stderr)
            sys.exit(1)
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        label = f"Groq ({model})"

    print(
        f"ERP bot ({label}). Datos: {data_backend_label()}.\n"
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
            if provider == "gemini":
                reply = run_turn_gemini(model, api_key, list(history))
            elif provider == "ollama":
                reply = run_turn_ollama(base_url, model, list(history))
            else:
                reply = run_turn(client, model, list(history))
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
