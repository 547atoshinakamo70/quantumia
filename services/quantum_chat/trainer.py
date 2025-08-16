import os, json, math, glob
import numpy as np
from typing import Dict, List
from .qnn32 import QNN32Ensemble

class QNNTrainer:
    """ Simple & safe: trains only on local files under data_dir (no network).

         Expects JSONL with fields {"text": str, "label": float in [0,1]} or CSV (text,label).
"""
    def __init__(self, ensemble:QNN32Ensemble, data_dir:str, out_path:str):
        self.ens=ensemble; self.data_dir=data_dir; self.out_path=out_path

    def _load(self)->List[Dict]:
        items=[]
        for p in glob.glob(os.path.join(self.data_dir, "**/*.jsonl"), recursive=True):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj=json.loads(line)
                        if isinstance(obj.get("text"), str) and isinstance(obj.get("label"),(int,float)):
                            items.append({"text": obj["text"], "label": float(obj["label"]) })
                    except Exception: pass
        for p in glob.glob(os.path.join(self.data_dir, "**/*.csv"), recursive=True):
            try:
                import csv
                with open(p, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        t=row.get("text") or row.get("content"); l=row.get("label") or row.get("y") or row.get("score")
                        if t is None or l is None: continue
                        items.append({"text": t, "label": float(l)})
            except Exception: pass
        return items

    def run(self, max_iters:int=10)->Dict:
        data=self._load()
        if not data:
            return {"ok": False, "msg":"No data found", "seen":0}
        rng=np.random.default_rng(0)
        # Toy 'training': nudges expert params towards matching labels via mean squared error gradient sign
        stats={"seen": len(data), "iters": max_iters}
        for it in range(max_iters):
            batch=rng.choice(data, size=min(64, len(data)), replace=False)
            loss=0.0
            for ex in batch:
                vec=self.ens.text_to_vec(ex["text"])
                pred=self.ens.predict(vec)["score"]
                y=ex["label"]; loss+=(pred-y)**2
                nud = 0.01 * (y - pred)
                self.ens.params += nud  # broadcasted small shift
            stats[f"loss_{it}"]= float(loss/len(batch))
        self.ens.save(self.out_path)
        return stats
