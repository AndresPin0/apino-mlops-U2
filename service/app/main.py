from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from datetime import datetime
import json
import os
from pathlib import Path

app = FastAPI(title="Demo Enfermedades – FastAPI")

STATS_FILE = Path(os.getenv("STATS_FILE_PATH", "/app/stats.json"))

def init_stats():
    """Inicializa el archivo de estadísticas si no existe"""
    if not STATS_FILE.exists():
        stats = {
            "total_by_category": {
                "NO ENFERMO": 0,
                "ENFERMEDAD LEVE": 0,
                "ENFERMEDAD AGUDA": 0,
                "ENFERMEDAD CRÓNICA": 0,
                "ENFERMEDAD TERMINAL": 0
            },
            "last_predictions": [],
            "last_prediction_date": None
        }
        save_stats(stats)
        return stats
    return load_stats()

def load_stats():
    """Carga las estadísticas desde el archivo"""
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return init_stats()

def save_stats(stats):
    """Guarda las estadísticas en el archivo"""
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

def update_stats(prediction: str):
    """Actualiza las estadísticas con una nueva predicción"""
    stats = load_stats()
    
    if prediction in stats["total_by_category"]:
        stats["total_by_category"][prediction] += 1
    else:
        stats["total_by_category"][prediction] = 1
    
    stats["last_predictions"].append({
        "status": prediction,
        "timestamp": datetime.now().isoformat()
    })
    if len(stats["last_predictions"]) > 5:
        stats["last_predictions"] = stats["last_predictions"][-5:]
    
    stats["last_prediction_date"] = datetime.now().isoformat()
    
    save_stats(stats)

LABELS = {
    0: "NO ENFERMO",
    1: "ENFERMEDAD LEVE",
    2: "ENFERMEDAD AGUDA",
    3: "ENFERMEDAD CRÓNICA",
    4: "ENFERMEDAD TERMINAL",
}

def rule_based_classifier(severity: float, duration_days: int, comorbidity_count: int) -> str:
    """
    Clasificador simple basado en reglas clínicas simuladas.
    Ahora incluye 5 categorías incluyendo ENFERMEDAD TERMINAL.
    """
    severity = max(0.0, min(10.0, float(severity)))
    duration_days = max(0, int(duration_days))
    comorbidity_count = max(0, int(comorbidity_count))

    if severity == 0 and duration_days == 0 and comorbidity_count == 0:
        return "NO ENFERMO"
    if (severity >= 9 and comorbidity_count >= 3) or \
       (severity >= 8 and duration_days >= 180 and comorbidity_count >= 2) or \
       (severity == 10):
        return "ENFERMEDAD TERMINAL"

    if duration_days >= 90 and comorbidity_count >= 2:
        return "ENFERMEDAD CRÓNICA"

    if severity >= 7 or (duration_days <= 14 and severity >= 5):
        return "ENFERMEDAD AGUDA"

    if 1 <= severity < 7 or (duration_days < 30 and comorbidity_count == 0):
        return "ENFERMEDAD LEVE"

    return "NO ENFERMO"


@app.get("/predict")
def predict(
    severity: float = Query(..., description="Severidad 0-10"),
    duration_days: int = Query(..., description="Duración en días (≥0)"),
    comorbidity_count: int = Query(..., description="Número de comorbilidades (≥0)")
):
    """Realiza una predicción del estado de enfermedad del paciente"""
    label = rule_based_classifier(severity, duration_days, comorbidity_count)
    
    update_stats(label)
    
    return JSONResponse({"status": label})


@app.get("/stats")
def get_stats():
    """Obtiene estadísticas de las predicciones realizadas"""
    stats = load_stats()
    
    return JSONResponse({
        "total_by_category": stats["total_by_category"],
        "last_5_predictions": stats["last_predictions"],
        "last_prediction_date": stats["last_prediction_date"]
    })


@app.get("/")
def root():
    """Endpoint raíz con información del servicio"""
    return JSONResponse({
        "service": "Demo Enfermedades - FastAPI",
        "endpoints": {
            "/predict": "Realiza una predicción (severity, duration_days, comorbidity_count)",
            "/stats": "Obtiene estadísticas de predicciones",
            "/": "Información del servicio"
        }
    })
