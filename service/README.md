# Servicio de predicción (demo) – FastAPI + Docker

Servicio de ejemplo que, a partir de 3 entradas (`severity`, `duration_days`, `comorbidity_count`),
retorna uno de: **NO ENFERMO | ENFERMEDAD LEVE | ENFERMEDAD AGUDA | ENFERMEDAD CRÓNICA**.

## Estructura
```
service/
├── app/
│   └── main.py
├── requirements.txt
├── MLops.postman_collection.json
└── Dockerfile
```

## Construir imagen
```bash
docker build -t demo-enfermedades:latest service
```

## Ejecutar contenedor
```bash
docker run --rm -p 8000:8000 demo-enfermedades:latest
```

## Probar por API (curl)
```bash
curl "http://localhost:8000/predict?severity=8&duration_days=3&comorbidity_count=0"
# Salida esperada:
# {"status":"ENFERMEDAD AGUDA"}
```