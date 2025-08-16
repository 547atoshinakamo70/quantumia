import os, time, math, json, hashlib, asyncio
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional research stack
try:
    import httpx
    from duckduckgo_search import DDGS
    import trafilatura
    import dateparser, tldextract
    from rapidfuzz import fuzz
    HAS_STACK = True
except Exception:
    HAS_STACK = False

# QNN32 integration
from .qnn32 import QNN32Ensemble, QNNConfig
from .trainer import QNNTrainer
from .crypto_utils import QuantumCrypto
from .assistant_proxy import ask_assistant

app = FastAPI(title="QuantumChat API (QNN32 + Research)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def inj(x:str)->bool:
    X=x.lower()
    bad=["rm -rf","shutdown","drop table","<?php","<script","curl http"]
    return any(b in X for b in bad)

def scrub(x:str)->str:
    return x.replace("\n"," ").strip()[:800]

def req_str(x,name,max_len=800)->str:
    if x is None: raise HTTPException(400,f"'{name}' required")
    if not isinstance(x,str): raise HTTPException(400,f"'{name}' must be string")
    x=x.strip();  x = x[:max_len]
    if not x: raise HTTPException(400,f"'{name}' empty")
    return x

DOMAIN_BOOST = {".gov":1.0,".edu":0.95,".org":0.75,"who.int":0.95}
BLOCKLIST = {"facebook.com","twitter.com","instagram.com","pinterest.","tiktok.","medium.com","quora.com"}

def tld_score(url:str)->float:
    if not HAS_STACK: return 0.5
    ex=tldextract.extract(url); host=f"{ex.domain}.{ex.suffix}".lower(); tld=f".{ex.suffix}".lower()
    s=0.5
    for k,v in DOMAIN_BOOST.items():
        if k in host or host.endswith(k) or k==tld: s=max(s,v)
    return s

def recency(ts:float)->float:
    if not ts: return 0.5
    days=max(0,(time.time()-ts)/86400.0)
    return 1.0/(1.0+math.exp((days-365)/180))

def extract_date(html:str)->float:
    if not HAS_STACK: return 0.0
    import re as _re
    metas=_re.findall(r'(?:date|time|published|updated)[^>]{0,120}content=[\'\"]([^\'\"]+)[\'\"]', html, flags=_re.I)
    cands=metas+_re.findall(r'(\d{4}-\d{2}-\d{2})', html)
    for c in cands[:5]:
        try:
            dt=dateparser.parse(c);  return dt.timestamp() if dt else 0.0
        except Exception: pass
    return 0.0

async def ddg(q:str, k:int=6)->List[Dict]:
    if not HAS_STACK: return []
    items=[]
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=k or 6):
                url=r.get("href") or r.get("url") or r.get("link")
                if not url: continue
                if any(b in url for b in BLOCKLIST): continue
                items.append({"title":r.get("title",url), "url":url})
    except Exception:
        return []
    return items

async def fetch(url:str)->Optional[Dict]:
    if not HAS_STACK: return None
    try:
        async with httpx.AsyncClient() as client:
            r=await client.get(url, timeout=8.0); r.raise_for_status()
            html=r.text; ts=extract_date(html)
            text = trafilatura.extract(html) or ""
            title = url
            import re as _re; m=_re.search(r"<title>(.*?)</title>", html, flags=_re.I|_re.S)
            if m: title = m.group(1).strip()
            return {"url":url,"title":title,"ts":ts,"text":text[:12000]}
    except Exception: return None

def content_score(doc:Dict,q:str)->float:
    if not HAS_STACK: return 0.6
    t=(doc.get("title","")+" "+doc.get("text",""))
    return min(1.0, (fuzz.token_sort_ratio(q.lower(),t.lower())/100.0)+0.15)

def final_rank(doc:Dict,q:str)->float:
    return 0.45*content_score(doc,q) + 0.30*tld_score(doc["url"]) + 0.25*recency(doc.get("ts",0.0))

def subqueries(q:str)->List[str]:
    out=[q]; L=q.lower()
    if "blockchain" in L or "bitcoin" in L: out += [q+" PoW difficulty", q+" mempool policies", q+" zk proofs best practices"]
    if "marketing" in L or "growth" in L: out += [q+" case study", q+" benchmark", q+" adoption 2025 trends"]
    return list(dict.fromkeys(out))[:6]

async def research(q:str, per_q:int=5, fetch_k:int=8)->Dict:
    subs=subqueries(q)
    items=[]
    for s in subs: items += await ddg(s, k=per_q)
    seen=set(); ranked=[]
    for it in items:
        url=it["url"]
        if url in seen: continue
        seen.add(url)
        doc=await fetch(url)
        if not doc: continue
        doc["score"]=final_rank(doc,q)
        ranked.append(doc)
    ranked.sort(key=lambda d: d.get("score",0), reverse=True)
    return {"subqueries":subs, "docs": ranked[:fetch_k]}

def concise(q:str, docs:List[Dict], n:int=6)->str:
    bullet=[f"• {d['title']} → {d['url']}" for d in docs[:n]]
    return "\n".join(bullet) or "Sin resultados relevantes."

# -------------------- QNN32 Ensemble state --------------------
QNN_PATH = os.environ.get("QNN32_PATH","./artifacts/qnn32_params.json")
os.makedirs(os.path.dirname(QNN_PATH), exist_ok=True)
qcfg = QNNConfig(n_qubits=4, depth=3, n_experts=32, shots=2048, seed=42)
ensemble = QNN32Ensemble(qcfg)

@app.post("/qnn32/predict")
async def qnn32_predict(body:Dict[str,Any]):
    payload = req_str(body.get("text"),"text",max_len=2000)
    # Featureize text → vector (toy: length & hash bucket)
    vec = ensemble.text_to_vec(payload)
    pred = ensemble.predict(vec)
    return JSONResponse({"ok":True, "vector": vec, "prediction": pred})

@app.post("/qnn32/train")  # trains only on local dataset (Quantum-Guard clone)
async def qnn32_train(body:Dict[str,Any]):
    data_dir = body.get("data_dir") or os.environ.get("QGUARD_DATA","./external/Quantum-Guard/data")
    trainer = QNNTrainer(ensemble, data_dir=data_dir, out_path=QNN_PATH)
    stats = trainer.run(max_iters=int(body.get("max_iters", 10)))
    return JSONResponse({"ok":True, "stats": stats, "saved": QNN_PATH})

@app.post("/research/complete")  # research + concise
async def complete(body:Dict[str,Any]):
    msg=req_str(body.get("message"),"message")
    if inj(msg): raise HTTPException(400,"Blocked by safety policy")
    msg=scrub(msg); n= int(body.get("bullets",6)) if str(body.get("bullets",""))
    else 6
    bundle=await research(msg, per_q=5, fetch_k=8)
    docs=bundle["docs"]; answer=concise(msg, docs, n)
    qc=QuantumCrypto()
    receipt=qc.create_receipt(hashlib.sha1((msg+str(time.time())).encode()).hexdigest()[:16],
                              [{"url":d["url"],"content":d.get("text","")[:400]} for d in docs], answer)
    return JSONResponse({"ok":True,"answer":answer,"sources":[{"url":d["url"],"title":d["title"]} for d in docs],
                         "subqueries":bundle["subqueries"], "receipt":receipt})

@app.post("/assistant/ask")  # Connect your tokenized assistant (this app → external LLM)
async def assistant(body:Dict[str,Any]):
    prompt=req_str(body.get("prompt"),"prompt",max_len=2000)
    model=os.environ.get("ASSISTANT_MODEL","gpt-4o-mini");
    resp=await ask_assistant(prompt, model=model)
    return JSONResponse({"ok":True, "reply": resp})
