import os, asyncio
import httpx

BASE=os.environ.get("ASSISTANT_BASE_URL","https://api.openai.com/v1")
TOKEN=os.environ.get("ASSISTANT_TOKEN",""")

async def ask_assistant(prompt:str, model:str="gpt-4o-mini") -> str:
    if not TOKEN:
        return "[assistant disabled: set ASSISTANT_TOKEN]"
    url=f"{BASE}/chat/completions"
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type":"application/json"}
    payload={"model": model, "messages": [{"role":"user","content":prompt}], "temperature":0.3}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r=await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        j=r.json()
        try:
            return j["choices"][0]["message"]["content"].strip()
        except Exception:
            return str(j)[:2000]
