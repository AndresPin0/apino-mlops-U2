# Repositorio MLOps - Predicción de Estados de Enfermedad

## Problema

Dados los avances tecnológicos en el campo de la medicina, la cantidad de información de pacientes es muy abundante. Sin embargo, para algunas enfermedades no tan comunes, llamadas huérfanas, los datos escasean. Se requiere construir un modelo que sea capaz de predecir, dados los datos de síntomas de un paciente, si es posible o no que este sufra de alguna enfermedad. Esto se requiere tanto para enfermedades comunes (muchos datos) como para enfermedades huérfanas (pocos datos).

## Propósito

Este repositorio contiene una solución completa de MLOps que incluye:

1. **Pipeline de MLOps**: Descripción completa del proceso end-to-end para el desarrollo del modelo (ver `pipeline/pipeline.md`)
2. **Servicio de Predicción**: Microservicio FastAPI contenedorizado con Docker que expone un modelo de predicción de estados de enfermedad
3. **CI/CD**: Pipeline automatizado con GitHub Actions para pruebas y despliegue

## Estructura del Repositorio

```
.
├── pipeline/
│   └── pipeline.md          # Descripción del pipeline MLOps
├── service/
│   ├── app/
│   │   └── main.py          # Aplicación FastAPI con el modelo
│   ├── tests/
│   │   └── test_model.py    # Pruebas unitarias
│   ├── Dockerfile           # Imagen Docker del servicio
│   ├── requirements.txt     # Dependencias Python
│   ├── MLops.postman_collection.json  # Colección Postman para pruebas
│   └── README.md            # Instrucciones del servicio
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # Pipeline CI/CD con GitHub Actions
└── README.md                # Este archivo
```

## Estados de Predicción

El modelo puede retornar uno de los siguientes estados:

- **NO ENFERMO**: Paciente sin síntomas o con valores en cero
- **ENFERMEDAD LEVE**: Síntomas suaves o moderados
- **ENFERMEDAD AGUDA**: Severidad alta o inicio reciente muy intenso
- **ENFERMEDAD CRÓNICA**: Duración larga con comorbilidades
- **ENFERMEDAD TERMINAL**: Condiciones críticas con múltiples factores de riesgo

## Uso del Servicio

### Construir la imagen Docker

```bash
docker build -t demo-enfermedades:latest service
```

### Ejecutar el contenedor

```bash
docker run --rm -p 8000:8000 demo-enfermedades:latest
```

### Realizar una predicción

```bash
curl "http://localhost:8000/predict?severity=8&duration_days=3&comorbidity_count=0"
```

### Obtener estadísticas

```bash
curl "http://localhost:8000/stats"
```

## API Endpoints

- `GET /predict`: Realiza una predicción basada en síntomas del paciente
  - Parámetros: `severity` (0-10), `duration_days` (≥0), `comorbidity_count` (≥0)
  - Retorna: `{"status": "ESTADO"}`

- `GET /stats`: Obtiene estadísticas de las predicciones realizadas
  - Retorna: Número total por categoría, últimas 5 predicciones, fecha de última predicción

## Desarrollo

### Ejecutar pruebas localmente

```bash
cd service
pip install -r requirements.txt
pytest tests/
```

## Detalles Técnicos

- **Framework**: FastAPI
- **Lenguaje**: Python 3.11
- **Contenedorización**: Docker
- **CI/CD**: GitHub Actions
- **Registro de imágenes**: GitHub Packages

## Autor

Andrés Pino
