import os, json, math, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

try:
    from qiskit import QuantumCircuit, Aer, execute
    HAS_QISKIT=True
except Exception:
    HAS_QISKIT=False

@dataclass
class QNNConfig:
    n_qubits:int=4
    depth:int=3
    n_experts:int=32
    shots:int=2048
    seed:int=42

class QNN32Ensemble:
    def __init__(self, cfg:QNNConfig):
        self.cfg=cfg
        self.params = np.random.default_rng(cfg.seed).normal(0, 0.5, size=(cfg.n_experts, cfg.depth, cfg.n_qubits, 2))
        self.backend = Aer.get_backend('qasm_simulator') if HAS_QISKIT else None

    def _circuit(self, theta:np.ndarray):
        qc = QuantumCircuit(self.cfg.n_qubits, self.cfg.n_qubits)
        for d in range(self.cfg.depth):
            for q in range(self.cfg.n_qubits):
                qc.ry(float(theta[d,q,0]), q); qc.rz(float(theta[d,q,1]), q)
            # simple ring entanglement
            for q in range(self.cfg.n_qubits-1):
                qc.cx(q, q+1)
            qc.cx(self.cfg.n_qubits-1, 0)
        qc.measure(range(self.cfg.n_qubits), range(self.cfg.n_qubits))
        return qc

    def _run(self, qc):
        if not HAS_QISKIT: 
            # fallback: deterministic pseudo-measure
            return {'0'*self.cfg.n_qubits: 1.0}
        job=execute(qc, self.backend, shots=self.cfg.shots, seed_simulator=self.cfg.seed, seed_transpiler=self.cfg.seed)
        res=job.result().get_counts()
        total=sum(res.values())
        return {k: v/total for k,v in res.items()}

    def text_to_vec(self, text:str)->List[float]:
        # very simple features: length, vowels ratio, consonants ratio, hash buckets
        L=len(text); vowels=sum(c.lower() in 'aeiou' for c in text); cons=sum(c.isalpha() for c in text)-vowels
        h=int(hashlib.sha1(text.encode()).hexdigest(),16)
        buckets=[(h>>i)&0xFF for i in range(0,32,8)]
        s=float(L)+1e-9
        return [L/1024.0, vowels/s, cons/s] + [b/255.0 for b in buckets]

    def _encode(self, vec:List[float])->np.ndarray:
        # map 7-dim vec to gate angles for each layer/qubit
        vec=np.array(vec, dtype=float); 
        angles = np.zeros((self.cfg.depth, self.cfg.n_qubits, 2))
        for d in range(self.cfg.depth):
            for q in range(self.cfg.n_qubits):
                angles[d,q,0] = 2*math.pi*((vec[(q+d)%len(vec)] + 0.1*d) % 1.0)
                angles[d,q,1] = 2*math.pi*((vec[(q*2+d)%len(vec)] + 0.07*q) % 1.0)
        return angles

    def predict(self, vec:List[float])->Dict[str,Any]:
        # Run all experts and average bit-probabilities to a score in [0,1]
        enc=self._encode(vec)
        scores=[]
        for e in range(self.cfg.n_experts):
            theta = enc + self.params[e]
            qc = self._circuit(theta)
            probs = self._run(qc)
            # convert bitstrings to scalar via Hamming weight / n_qubits
            exp = 0.0; 
            for bits,p in probs.items():
                ones = bits.count('1')
                exp += p * (ones/self.cfg.n_qubits)
            scores.append(exp)
        score=float(np.mean(scores))
        return {"score": round(score,6), "by_expert": [round(float(s),6) for s in scores]}

    def save(self, path:str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.params.tolist(), f)

    def load(self, path:str):
        with open(path,"r",encoding="utf-8") as f:
            self.params = np.array(json.load(f), dtype=float)
