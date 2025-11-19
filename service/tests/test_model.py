"""
Pruebas unitarias para el modelo de predicción de enfermedades
"""
import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import sys
import os

_test_stats_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
_test_stats_file.close()
os.environ["STATS_FILE_PATH"] = _test_stats_file.name

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import app, rule_based_classifier, init_stats, load_stats, save_stats, update_stats, STATS_FILE

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_stats():
    """Fixture que limpia las estadísticas antes de cada test"""
    if Path(_test_stats_file.name).exists():
        Path(_test_stats_file.name).unlink()
    init_stats()
    yield


class TestModelPredictions:
    """Pruebas para las predicciones del modelo"""
    
    def test_prediction_enfermedad_leve(self):
        """
        Test 1: Dados unos parámetros de entrada, probar que la respuesta 
        del modelo es ENFERMEDAD LEVE esperado.
        Ejemplo: paciente con 20 años (simulado con síntomas leves), 
        síntomas leves respiratorios (severity=3) y alguna condición neurológica (comorbidity_count=0)
        """
        response = client.get("/predict?severity=3&duration_days=5&comorbidity_count=0")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ENFERMEDAD LEVE"
    
    def test_prediction_enfermedad_aguda(self):
        """Test: Verificar que severity alta retorna ENFERMEDAD AGUDA"""
        response = client.get("/predict?severity=8&duration_days=3&comorbidity_count=0")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ENFERMEDAD AGUDA"
    
    def test_prediction_no_enfermo(self):
        """Test: Verificar que sin síntomas retorna NO ENFERMO"""
        response = client.get("/predict?severity=0&duration_days=0&comorbidity_count=0")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NO ENFERMO"
    
    def test_prediction_enfermedad_cronica(self):
        """Test: Verificar que duración larga con comorbilidades retorna ENFERMEDAD CRÓNICA"""
        response = client.get("/predict?severity=5&duration_days=100&comorbidity_count=2")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ENFERMEDAD CRÓNICA"
    
    def test_prediction_enfermedad_terminal(self):
        """Test: Verificar que condiciones críticas retornan ENFERMEDAD TERMINAL"""
        response = client.get("/predict?severity=10&duration_days=10&comorbidity_count=1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ENFERMEDAD TERMINAL"
        
        response = client.get("/predict?severity=9&duration_days=50&comorbidity_count=3")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ENFERMEDAD TERMINAL"
    
    def test_all_5_categories_obtainable(self):
        """
        Test 2: Considerar distintos grupos de parámetros de entrada al modelo 
        y esperar que las 5 categorías de enfermedades sean obtenidas.
        """
        categories = set()
        
        # NO ENFERMO
        response = client.get("/predict?severity=0&duration_days=0&comorbidity_count=0")
        categories.add(response.json()["status"])
        
        # ENFERMEDAD LEVE
        response = client.get("/predict?severity=3&duration_days=5&comorbidity_count=0")
        categories.add(response.json()["status"])
        
        # ENFERMEDAD AGUDA
        response = client.get("/predict?severity=8&duration_days=3&comorbidity_count=0")
        categories.add(response.json()["status"])
        
        # ENFERMEDAD CRÓNICA
        response = client.get("/predict?severity=5&duration_days=100&comorbidity_count=2")
        categories.add(response.json()["status"])
        
        # ENFERMEDAD TERMINAL
        response = client.get("/predict?severity=10&duration_days=10&comorbidity_count=1")
        categories.add(response.json()["status"])
        
        # Verificar que se obtuvieron las 5 categorías
        expected_categories = {
            "NO ENFERMO",
            "ENFERMEDAD LEVE",
            "ENFERMEDAD AGUDA",
            "ENFERMEDAD CRÓNICA",
            "ENFERMEDAD TERMINAL"
        }
        assert categories == expected_categories, f"Se esperaban 5 categorías, se obtuvieron: {categories}"


class TestStatistics:
    """Pruebas para las estadísticas de predicciones"""
    
    def test_stats_initial_empty(self):
        """
        Test: Antes de correr cualquier predicción, pedir las estadísticas 
        y esperar que estas se encuentren vacías o con los valores por defecto.
        """
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        
        # Verificar estructura
        assert "total_by_category" in data
        assert "last_5_predictions" in data
        assert "last_prediction_date" in data
        
        # Verificar que todas las categorías están presentes
        assert "NO ENFERMO" in data["total_by_category"]
        assert "ENFERMEDAD LEVE" in data["total_by_category"]
        assert "ENFERMEDAD AGUDA" in data["total_by_category"]
        assert "ENFERMEDAD CRÓNICA" in data["total_by_category"]
        assert "ENFERMEDAD TERMINAL" in data["total_by_category"]
        
        # Verificar valores iniciales (todos en 0, lista vacía, fecha None)
        assert sum(data["total_by_category"].values()) == 0
        assert len(data["last_5_predictions"]) == 0
        assert data["last_prediction_date"] is None
    
    def test_stats_updated_after_prediction(self):
        """
        Test: Realizar una predicción que arroje algún tipo de enfermedad, 
        y luego chequear las estadísticas para asegurarse que la última 
        predicción realizada sea la esperada.
        """
        # Obtener estadísticas iniciales
        initial_response = client.get("/stats")
        initial_data = initial_response.json()
        initial_count = initial_data["total_by_category"]["ENFERMEDAD LEVE"]
        
        # Realizar una predicción
        prediction_response = client.get("/predict?severity=3&duration_days=5&comorbidity_count=0")
        assert prediction_response.status_code == 200
        prediction_data = prediction_response.json()
        predicted_status = prediction_data["status"]
        
        # Obtener estadísticas después de la predicción
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        
        # Verificar que el conteo aumentó
        assert stats_data["total_by_category"][predicted_status] == initial_count + 1
        
        # Verificar que la última predicción es la esperada
        assert len(stats_data["last_5_predictions"]) > 0
        last_prediction = stats_data["last_5_predictions"][-1]
        assert last_prediction["status"] == predicted_status
        assert "timestamp" in last_prediction
        
        # Verificar que la fecha de última predicción se actualizó
        assert stats_data["last_prediction_date"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
