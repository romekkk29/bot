"""
Mide el costo real en tokens de los schemas de tools.
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types as gt
from tools import TOOLS

load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]
model = "gemini-2.0-flash"
client = genai.Client(api_key=api_key)

tool = gt.Tool(
    function_declarations=[
        gt.FunctionDeclaration(
            name=t["function"]["name"],
            description=t["function"]["description"],
        )
        for t in TOOLS
    ]
)

msg = [gt.Content(role="user", parts=[gt.Part.from_text(text="hola")])]

cfg_sin = gt.GenerateContentConfig(temperature=0)
cfg_con = gt.GenerateContentConfig(temperature=0, tools=[tool])

r_sin = client.models.generate_content(model=model, contents=msg, config=cfg_sin)
r_con = client.models.generate_content(model=model, contents=msg, config=cfg_con)

u_sin = r_sin.usage_metadata
u_con = r_con.usage_metadata

schemas_tokens = u_con.prompt_token_count - u_sin.prompt_token_count

print(f"Mensaje solo  (sin tools): {u_sin.prompt_token_count} tokens input")
print(f"Mensaje + {len(TOOLS)} tools:       {u_con.prompt_token_count} tokens input")
print(f"Costo de schemas solos:    {schemas_tokens} tokens")
print(f"Costo por tool (promedio): {schemas_tokens // len(TOOLS)} tokens/tool")