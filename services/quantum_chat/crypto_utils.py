import hashlib, time
class QuantumCrypto:
    def create_receipt(self, root:str, docs, answer:str)->str:
        h=hashlib.sha256()
        h.update(root.encode())
        for d in docs:
            h.update((d.get("url","")+d.get("content",""))[:512].encode())
        h.update(answer[:1024].encode())
        h.update(str(int(time.time())).encode())
        return h.hexdigest()
