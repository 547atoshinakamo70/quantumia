# Integración QNN32 ↔ Quantum-Guard ↔ Assistant (token)

Este paquete añade a tu proyecto:
- **QNN32 (32 redes neuronales cuánticas)** con Qiskit (ensamble).
- **Entrenamiento local** solo con datos de *Quantum-Guard* (no hay fetch por red).
- **Endpoints FastAPI** para predecir/entrenar.
- **Proxy a tu asistente por token** (conecta este servicio a tu LLM favorito).

## Estructura
```
services/quantum_chat/
  ├─ quantum_agent.py        # FastAPI con /qnn32/*, /research/* y /assistant/ask
  ├─ qnn32.py                # implementación QNN32 (Qiskit)
  ├─ trainer.py              # entrenamiento simple offline
  ├─ assistant_proxy.py      # proxy a Chat API vía token (configurable)
  ├─ crypto_utils.py         # recibo/verificabilidad simple
  └─ requirements.txt
client/
  └─ src/pages/QNN32.tsx     # página de prueba (si usas tu PWA)
```

## Endpoints
- `POST /qnn32/predict`  body: `{ "text": "..." }` → `{ score, by_expert[] }`
- `POST /qnn32/train`    body: `{ "data_dir":"./external/Quantum-Guard/data", "max_iters": 10 }`
- `POST /research/complete`  body: `{ "message":"..." }` → resumen con fuentes
- `POST /assistant/ask`  body: `{ "prompt":"..." }` → respuesta del asistente conectado por token

## Conectar **Quantum-Guard** (solo como dataset)
1) Añade el repo como submódulo (ajusta la URL real):
```bash
git submodule add https://github.com/ZAKIBAYDOUN/Quantum-Gaurd external/Quantum-Guard
# si el nombre correcto es Quantum-Guard (con 'u'), usa la URL válida
```
2) Asegúrate de tener datos en `external/Quantum-Guard/data/` como:
   - JSONL con líneas `{"text": "...", "label": 0..1}`
   - o CSV con columnas `text,label`

> *Este trainer NO hace red; solo lee archivos locales.*

## Conectar tu asistente por **token**
Exporta variables:
```bash
export ASSISTANT_BASE_URL=https://api.openai.com/v1
export ASSISTANT_TOKEN=sk-...   # tu token
export ASSISTANT_MODEL=gpt-4o-mini
```
Luego:
```bash
curl -X POST http://127.0.0.1:5002/assistant/ask -H 'Content-Type: application/json'   -d '{"prompt":"Evalúa el siguiente roadmap..."}'
```

## Ejecutar API
```bash
cd services/quantum_chat
uvicorn quantum_agent:app --host 0.0.0.0 --port 5002
```
(Instala requisitos: `pip install -r requirements.txt`)

## Nota de entrenamiento
El trainer es ligero (demostrativo). Para producción:
- Persistencia de parámetros por expero.
- Scheduler/colas (ej. Redis) para tareas.
- Métricas (p.ej., pérdida por época, score holdout).
- Mejor _feature engineering_ para `text_to_vec` o vectorizador semántico (sin salirte del dataset local).
