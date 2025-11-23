# MLOps Pipeline para Predicción de Enfermedades (Comunes y Huérfanas) → V2.0.0

## Caso de Uso

### Contexto

En el campo de la medicina moderna, existe una abundancia de datos para enfermedades comunes (diabetes, hipertensión, neumonía), pero una escasez crítica de datos para enfermedades huérfanas o raras (menos de 5 casos por 10,000 personas). Esta asimetría plantea desafíos únicos para el desarrollo de modelos de machine learning que puedan predecir ambos tipos de enfermedades de manera efectiva.

### Definición del Problema

**Objetivo:** Construir un sistema MLOps end-to-end que permita predecir, dados los síntomas y datos clínicos de un paciente, si este puede estar sufriendo de alguna enfermedad (común o huérfana), y clasificar la severidad del estado del paciente.

**Usuarios finales:** Médicos que pueden usar el modelo de dos formas:
1. **Local:** Si los recursos computacionales son bajos, pueden ejecutar el modelo en su computador
2. **Remoto:** Pueden hacer peticiones HTTP a un modelo alojado en la nube/servidor

**Restricciones clave:**
- Privacidad de datos médicos (cumplimiento HIPAA/GDPR)
- Explicabilidad de las predicciones para uso clínico
- Manejo de datos desbalanceados (enfermedades huérfanas)
- Baja latencia para uso en entornos clínicos
- Trazabilidad completa de predicciones

---

## Arquitectura del Pipeline MLOps

### Diagrama 1: Arquitectura Completa del Sistema

![Diagrama de arquitectura del pipeline MLOps](/pipeline/diagramas/arquitectura_sistema.png)

## Descripción Detallada de Cada Etapa

### 1. Control de Versiones y CI/CD

**Tecnologías:**
- **GitHub**: Repositorio de código fuente
- **GitHub Actions**: Pipeline CI/CD

**Justificación:**
GitHub es el estándar de la industria para control de versiones. GitHub Actions permite automatizar testing, validación y deployment sin necesidad de servicios externos adicionales, reduciendo la complejidad operacional.

**Implementación:**
- Cada cambio en el código activa:
  - Linting (flake8, black)
  - Unit tests (pytest)
  - Integration tests
  - Security scanning (Snyk, Bandit)
- Branch protection rules: requiere aprobación de PR y tests pasando
- Versionamiento semántico (SemVer) para releases

**Suposiciones:**
- El equipo tiene acceso a GitHub y puede usar GitHub Actions
- Se implementan revisiones de código por pares

---

### 2. Ingesta y Almacenamiento de Datos

#### 2.1 Fuentes de Datos

**Datos primarios:**
- **Registros Médicos Electrónicos (EMR/HIS)**: historias clínicas en formatos HL7 o FHIR
- **Sistemas de Información de Laboratorio (LIS)**: resultados de laboratorio
- **Formularios de admisión**: síntomas reportados por pacientes
- **Registros de farmacias**: medicamentos prescritos

**Datos externos:**
- APIs de laboratorios de referencia
- Bases de datos genómicas (para enfermedades hereditarias)
- Registros de enfermedades raras (ORPHANET, NORD)

#### 2.2 Almacenamiento Raw

**Tecnología:** AWS S3 (Simple Storage Service)

**Justificación:**
- Escalable: maneja desde KB hasta PB de datos
- Durabilidad: 99.999999999% (11 nueves)
- Compatible con herramientas de data science (boto3, pandas)
- Bajo costo para almacenamiento de datos históricos
- Soporte nativo para encriptación (SSE-S3, SSE-KMS) cumpliendo HIPAA

**Implementación:**
- Estructura de buckets:
  ```
  s3://medical-data-prod/
    ├── raw/              # Datos sin procesar
    │   ├── emr/
    │   ├── lab_results/
    │   └── external/
    ├── processed/        # Datos transformados
    └── features/         # Features engineered
  ```
- Lifecycle policies: mover datos antiguos a S3 Glacier después de 1 año
- Versionado habilitado para auditoría
- Access logs habilitados

**Suposiciones:**
- Los datos llegan en formatos semi-estructurados (JSON, CSV, XML)
- La infraestructura está en AWS (o se migrará)
- Los hospitales/clínicas pueden enviar datos a través de APIs seguras o uploads

#### 2.3 Preprocesamiento y Versionado

**Tecnología:** DVC (Data Version Control)

**Justificación:**
DVC permite versionar datasets grandes (que Git no puede manejar eficientemente) y mantener trazabilidad entre código, datos y modelos. Esencial para reproducibilidad científica.

**Implementación:**
```bash
# Versionar dataset
dvc add data/training_dataset_v1.csv
git add data/training_dataset_v1.csv.dvc
git commit -m "Add training dataset v1"
git tag -a "data-v1" -m "Dataset version 1"
```

**Procesamiento:**
1. **Validación de esquema**: Usar Pydantic/Pandera para validar tipos de datos
2. **Anonimización**: Eliminar/encriptar PII/PHI (nombres, direcciones, IDs)
3. **Normalización**: Convertir unidades (mg/dL vs mmol/L), estandarizar códigos de diagnóstico (ICD-10)
4. **Limpieza**: Manejo de valores faltantes, outliers

**Suposiciones:**
- Los datos pueden contener errores de entrada manual
- Es necesario mantener múltiples versiones de datasets para diferentes experimentos

#### 2.4 Data Warehouse

**Tecnología:** Snowflake o Amazon Redshift

**Justificación:**

**Snowflake (Preferido):**
- Separación de cómputo y almacenamiento: escala independientemente
- Soporte nativo para datos semi-estructurados (JSON, XML) común en medicina
- Data sharing seguro entre instituciones
- Compliant con HIPAA desde el inicio
- Time travel: recuperar datos históricos

**Redshift (Alternativa):**
- Integración nativa con AWS (S3, SageMaker)
- Menor costo si ya se usa AWS
- Buen rendimiento para queries analíticas

**Implementación:**
```sql
-- Tabla de pacientes (anonimizada)
CREATE TABLE patients (
    patient_id VARCHAR(64) PRIMARY KEY,  -- Hash del ID real
    age INT,
    gender VARCHAR(10),
    comorbidities ARRAY,
    admission_date TIMESTAMP
);

-- Tabla de síntomas
CREATE TABLE symptoms (
    symptom_id VARCHAR(64) PRIMARY KEY,
    patient_id VARCHAR(64),
    symptom_code VARCHAR(20),  -- SNOMED CT code
    severity INT,
    duration_days INT,
    timestamp TIMESTAMP
);

-- Tabla de diagnósticos (ground truth)
CREATE TABLE diagnoses (
    diagnosis_id VARCHAR(64) PRIMARY KEY,
    patient_id VARCHAR(64),
    disease_code VARCHAR(20),  -- ICD-10 code
    disease_category VARCHAR(50),  -- COMMON/RARE
    confirmed_date TIMESTAMP
);
```

**Suposiciones:**
- Se necesita realizar queries complejas sobre datos históricos
- Múltiples usuarios (data scientists, analistas) necesitan acceso concurrente

---

### 3. Iteraciones del Modelo

#### 3.1 Exploración y Análisis de Datos (EDA)

**Tecnologías:**
- **Jupyter Notebooks**: Entorno interactivo
- **Pandas, NumPy**: Manipulación de datos
- **Matplotlib, Seaborn, Plotly**: Visualización
- **AWS EC2 (instancias m5.xlarge o superiores)**: Cómputo para datasets grandes

**Justificación:**
Jupyter es el estándar de facto para data science. EC2 permite escalar recursos según necesidad (instancias grandes para EDA intensivo, apagar cuando no se usa).

**Actividades:**
1. **Análisis de distribuciones**: ¿Cómo se distribuyen las enfermedades? (esperamos heavy imbalance)
2. **Análisis de missingness**: ¿Qué features tienen valores faltantes y por qué?
3. **Correlaciones**: ¿Qué síntomas están correlacionados con qué enfermedades?
4. **Análisis temporal**: ¿Hay tendencias estacionales?
5. **Análisis de desbalance**: Calcular ratio common/rare diseases

**Hallazgos esperados:**
- Ratio de 1:1000 o peor para enfermedades huérfanas
- Muchos valores faltantes en datos de laboratorio (no siempre se piden todos los exámenes)
- Alta correlación entre ciertos síntomas y enfermedades específicas

**Suposiciones:**
- Los data scientists tienen experiencia en análisis de datos médicos
- Los datos de enfermedades raras serán extremadamente escasos (<100 casos por enfermedad)

#### 3.2 Feature Engineering

**Objetivo:** Crear features informativas que mejoren la capacidad predictiva del modelo.

**Features a crear:**

1. **Features de agregación:**
   - Número total de síntomas
   - Severidad promedio de síntomas
   - Duración máxima de síntomas
   - Número de comorbilidades

2. **Features temporales:**
   - Velocidad de progresión (cambio de severidad en el tiempo)
   - Estacionalidad (mes, estación del año)

3. **Features de interacción:**
   - Combinaciones de síntomas que suelen aparecer juntas
   - Edad × severidad
   - Comorbilidades × síntomas

4. **Features basadas en conocimiento médico:**
   - Scores clínicos existentes (ej. Apache II, SOFA score)
   - Similitud a casos conocidos (embeddings)

5. **Features de texto (si hay notas clínicas):**
   - TF-IDF de términos médicos
   - Embeddings de BERT médico (BioBERT, ClinicalBERT)

**Tecnologías:**
- **Featuretools**: Automated feature engineering
- **scikit-learn**: Transformers personalizados
- **Hugging Face Transformers**: Para embeddings de texto médico

**Implementación:**
```python
from featuretools import dfs

# Automated feature engineering
feature_matrix, feature_defs = dfs(
    entityset=medical_entityset,
    target_dataframe_name='patients',
    max_depth=2,
    verbose=True
)
```

**Suposiciones:**
- Las features derivadas pueden mejorar la precisión en 5-15%
- No todas las features serán útiles (necesitaremos feature selection)

#### 3.3 Manejo de Datos Desbalanceados

**Problema crítico:** Las enfermedades raras representan <0.1% de los casos.

**Estrategias:**

1. **Para enfermedades comunes:**
   - **SMOTE (Synthetic Minority Over-sampling)**: Generar casos sintéticos
   - **Class weighting**: Penalizar más los errores en la clase minoritaria
   - **Undersampling de la clase mayoritaria**: Reducir casos de "no enfermo"

2. **Para enfermedades raras:**
   - **Few-shot learning**: Entrenar con pocos ejemplos
   - **Transfer learning**: Pre-entrenar en enfermedades comunes, fine-tune en raras
   - **Meta-learning (MAML)**: Aprender a aprender rápido con pocos datos
   - **Data augmentation**: Generar variaciones de los pocos casos existentes
   - **One-class classification**: Detectar anomalías vs. casos normales

**Tecnologías:**
- **imbalanced-learn**: Librería especializada en datos desbalanceados
- **PyTorch Meta-Learning**: Para meta-learning
- **Siamese Networks**: Para few-shot learning

**Justificación:**
No hay una solución única para datos extremadamente desbalanceados. Necesitamos una combinación de técnicas. Transfer learning es particularmente prometedor en medicina porque las enfermedades comparten síntomas.

**Suposiciones:**
- Las enfermedades raras comparten algunos síntomas con enfermedades comunes
- Los médicos pueden proporcionar feedback para validar casos sintéticos

#### 3.4 Entrenamiento de Modelos

**Enfoque dual:** Dos tipos de modelos según disponibilidad de datos.

##### Modelos para Enfermedades Comunes (>1000 casos)

**Algoritmos:**
1. **XGBoost**: Excelente para datos tabulares, maneja missing values
2. **Random Forest**: Robusto, proporciona feature importance
3. **LightGBM**: Más rápido que XGBoost, similar performance
4. **Neural Networks**: MLP o TabNet para capturar relaciones no lineales complejas

**Tecnologías:**
- **XGBoost, LightGBM**: Librerías especializadas
- **PyTorch/TensorFlow**: Para redes neuronales
- **Optuna**: Hyperparameter tuning automático
- **MLflow**: Tracking de experimentos

**Configuración de ejemplo (XGBoost):**
```python
import xgboost as xgb
import optuna

def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100)
    }
    
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Tracking con MLflow:**
```python
import mlflow
import mlflow.xgboost

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(best_params)
    
    # Train model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Log metrics
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
    
    # Log model
    mlflow.xgboost.log_model(model, "model")
```

##### Modelos para Enfermedades Raras (<100 casos)

**Estrategia 1: Few-Shot Learning**

Entrenar una red neuronal que aprende embeddings donde enfermedades similares están cercanas en el espacio latente.

```python
# Siamese Network para few-shot learning
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    def compute_distance(self, x1, x2):
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        return torch.norm(emb1 - emb2, dim=1)
```

**Estrategia 2: Transfer Learning**

Pre-entrenar en enfermedades comunes, luego fine-tune en raras:

```python
# Pre-train en enfermedades comunes
base_model = train_on_common_diseases(X_common, y_common)

# Fine-tune en enfermedades raras
rare_model = fine_tune(base_model, X_rare, y_rare, 
                        learning_rate=1e-5, epochs=50)
```

**Estrategia 3: Ensemble con Conocimiento Médico**

Combinar ML con reglas basadas en conocimiento médico:

```python
def hybrid_prediction(ml_model, rules_engine, patient_data):
    ml_score = ml_model.predict_proba(patient_data)
    rule_score = rules_engine.evaluate(patient_data)
    
    # Weighted ensemble
    final_score = 0.7 * ml_score + 0.3 * rule_score
    return final_score
```

**Tecnologías:**
- **PyTorch**: Para implementar arquitecturas personalizadas
- **Sentence-BERT**: Para embeddings si usamos texto
- **Optuna**: Para optimizar arquitecturas y hyperparámetros

**Justificación:**
Las enfermedades raras requieren técnicas especializadas porque los métodos tradicionales fallan con pocos datos. Few-shot learning y transfer learning han demostrado éxito en problemas similares (reconocimiento de imágenes raras, NLP con pocas muestras).

**Suposiciones:**
- Existe suficiente data de enfermedades comunes para pre-entrenar (>10,000 casos)
- Las enfermedades raras comparten features con las comunes
- Los médicos pueden proporcionar reglas basadas en conocimiento

#### 3.5 Calibración de Modelos

**Problema:** Los modelos ML no siempre producen probabilidades bien calibradas. Si el modelo dice "90% de probabilidad de enfermedad X", ¿realmente 90% de esos casos tienen X?

**Técnicas:**
- **Platt Scaling**: Regresión logística sobre las probabilidades del modelo
- **Isotonic Regression**: Más flexible que Platt, pero requiere más datos
- **Temperature Scaling**: Para redes neuronales

**Implementación:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar modelo base
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Evaluar calibración
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_test, 
    calibrated_model.predict_proba(X_test)[:, 1],
    n_bins=10
)
```

**Métricas de calibración:**
- **Expected Calibration Error (ECE)**: Diferencia entre probabilidad predicha y frecuencia observada
- **Brier Score**: MSE de las probabilidades

**Justificación:**
En medicina, la calibración es crítica. Un médico necesita confiar en que una probabilidad del 80% realmente significa que 8 de cada 10 pacientes tienen la enfermedad.

**Suposiciones:**
- Tenemos suficiente data de validación para calibrar (>500 casos)
- Las probabilidades calibradas se usarán para decisiones clínicas

---

### 4. Selección y Evaluación del Modelo

#### 4.1 Métricas de Evaluación

**Para clasificación desbalanceada:**

1. **AUPRC (Area Under Precision-Recall Curve)**: Más informativo que AUROC cuando hay desbalance
2. **AUROC (Area Under ROC Curve)**: Métrica estándar
3. **F1-Score**: Balance entre precisión y recall
4. **Sensibilidad (Recall)**: Crítico en medicina - no queremos perder casos verdaderos
5. **Especificidad**: Importante para no alarmar innecesariamente
6. **Expected Calibration Error (ECE)**: Mide calibración

**Para enfermedades raras específicamente:**
- **Recall en clase rara**: ¿Detectamos los pocos casos raros?
- **False Negative Rate**: Costo muy alto en medicina

**Implementación:**
```python
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    classification_report,
    confusion_matrix
)

# Calcular métricas
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

metrics = {
    'auroc': roc_auc_score(y_test, y_pred_proba),
    'auprc': average_precision_score(y_test, y_pred_proba),
    'f1': f1_score(y_test, y_pred),
    'sensitivity': recall_score(y_test, y_pred),
    'specificity': recall_score(1 - y_test, 1 - y_pred),
}

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**Umbrales personalizados:**
```python
# Optimizar umbral para sensibilidad mínima del 95%
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Encontrar umbral que da recall >= 0.95
target_recall = 0.95
idx = np.argmax(recalls >= target_recall)
optimal_threshold = thresholds[idx]

print(f"Threshold for 95% recall: {optimal_threshold}")
print(f"Precision at this threshold: {precisions[idx]}")
```

**Justificación:**
En medicina, los falsos negativos (no detectar una enfermedad) suelen ser más costosos que los falsos positivos (tests adicionales). Por eso priorizamos sensibilidad.

#### 4.2 Validación Cruzada

**Estrategia:** Stratified K-Fold para mantener proporciones de clases

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    
    model.fit(X_train_cv, y_train_cv)
    score = roc_auc_score(y_val_cv, model.predict_proba(X_val_cv)[:, 1])
    cv_scores.append(score)

print(f"CV AUROC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

**Validación temporal:**
Para evitar data leakage, también validamos con split temporal:
- Train: 2020-2022
- Validation: 2023 (Q1-Q3)
- Test: 2023 (Q4)

**Justificación:**
Validación temporal asegura que el modelo funciona en datos futuros, no solo en datos contemporáneos al entrenamiento.

**Suposiciones:**
- Los datos tienen timestamps confiables
- La distribución de enfermedades no cambia drásticamente año a año

#### 4.3 Explicabilidad

**Técnicas:**

1. **SHAP (SHapley Additive exPlanations)**: Explica contribución de cada feature

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizar importancia de features
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Explicar predicción individual
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
```

2. **LIME (Local Interpretable Model-agnostic Explanations)**: Explica predicciones locales

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, 
    feature_names=feature_names,
    class_names=['No disease', 'Disease'],
    mode='classification'
)

# Explicar una predicción
exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
```

3. **Feature Importance**: Para modelos tree-based

```python
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(20), feature_importance[sorted_idx][:20])
plt.xticks(range(20), [feature_names[i] for i in sorted_idx[:20]], rotation=45)
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

**Tecnologías:**
- **SHAP**: Librería especializada en explicabilidad
- **LIME**: Alternativa más rápida pero menos precisa
- **Integrated Gradients**: Para redes neuronales

**Justificación:**
La explicabilidad es CRÍTICA en medicina. Los médicos no usarán una "caja negra". Necesitan entender por qué el modelo hizo una predicción para validarla clínicamente y tomar decisiones informadas.

**Suposiciones:**
- Los médicos tienen tiempo para revisar explicaciones
- Las explicaciones son comprensibles para no-expertos en ML

#### 4.4 Visualización para Stakeholders

**Tecnología:** Streamlit

**Justificación:**
Streamlit permite crear dashboards interactivos con pocas líneas de código Python, perfecto para mostrar resultados a médicos y administradores sin experiencia técnica.

**Implementación:**
```python
import streamlit as st
import plotly.express as px

st.title('Disease Prediction Model Dashboard')

# Métricas principales
col1, col2, col3 = st.columns(3)
col1.metric("AUROC", f"{metrics['auroc']:.3f}")
col2.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
col3.metric("Specificity", f"{metrics['specificity']:.3f}")

# Confusion matrix
st.subheader('Confusion Matrix')
fig = px.imshow(confusion_matrix(y_test, y_pred),
                labels=dict(x="Predicted", y="Actual"),
                x=['No Disease', 'Disease'],
                y=['No Disease', 'Disease'])
st.plotly_chart(fig)

# ROC Curve
st.subheader('ROC Curve')
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig = px.line(x=fpr, y=tpr, title='ROC Curve')
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
st.plotly_chart(fig)

# Feature importance
st.subheader('Feature Importance')
fig = px.bar(x=feature_names[:20], y=feature_importance[:20])
st.plotly_chart(fig)
```

**Suposiciones:**
- Los stakeholders tienen acceso a un navegador web
- El dashboard puede actualizarse semanalmente (no necesita ser tiempo real)

#### 4.5 Criterios de Aceptación

**El modelo pasa a producción si cumple:**

1. **Performance mínimo:**
   - AUROC ≥ 0.85 para enfermedades comunes
   - AUROC ≥ 0.75 para enfermedades raras
   - Sensibilidad ≥ 90% para enfermedades graves
   - ECE ≤ 0.1 (buena calibración)

2. **Validación clínica:**
   - Aprobación de comité médico
   - Revisión de al menos 100 predicciones por médicos expertos
   - Casos de error analizados y documentados

3. **Performance técnico:**
   - Latencia ≤ 500ms para predicción individual
   - Throughput ≥ 100 predicciones/segundo para batch
   - Tamaño del modelo ≤ 500MB (para deployment local)

4. **Explicabilidad:**
   - SHAP values calculables en <2 segundos
   - Top 5 features más importantes son clínicamente interpretables

5. **Robustez:**
   - Performance degradada ≤ 5% con 20% de missing values
   - Performance estable en diferentes subgrupos (edad, género, etnicidad)

**Suposiciones:**
- El comité médico está disponible para revisiones
- Los criterios se ajustarán basándose en feedback inicial

---

### 5. Deployment

#### 5.1 Serialización del Modelo

**Tecnologías:**
- **MLflow Model Registry**: Versionamiento y staging de modelos
- **Pickle/Joblib**: Serialización de modelos scikit-learn
- **ONNX**: Formato interoperable para redes neuronales
- **TorchScript**: Para modelos PyTorch

**Implementación:**
```python
import mlflow
import mlflow.sklearn

# Registrar modelo en MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model, 
        "disease_prediction_model",
        registered_model_name="DiseasePredictor"
    )

# Transicionar a staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="DiseasePredictor",
    version=1,
    stage="Staging"
)

# Después de validación, promover a production
client.transition_model_version_stage(
    name="DiseasePredictor",
    version=1,
    stage="Production"
)
```

**Justificación:**
MLflow Model Registry proporciona versionamiento, staging (development → staging → production), y trazabilidad completa. Esencial para ambientes regulados como medicina.

#### 5.2 Containerización

**Tecnología:** Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelo
COPY app/ ./app/
COPY models/ ./models/

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
xgboost==2.0.2
numpy==1.26.2
pandas==2.1.3
mlflow==2.8.1
prometheus-client==0.19.0  # Para métricas
```

**Justificación:**
Docker garantiza que el modelo funciona idénticamente en desarrollo, staging y producción. Crítico para reproducibilidad y evitar el clásico "en mi máquina funciona".

#### 5.3 API de Predicción

**Tecnología:** FastAPI

**Justificación:**
- Más rápido que Flask (basado en Starlette + asyncio)
- Validación automática con Pydantic
- Documentación automática (OpenAPI/Swagger)
- Type hints nativos de Python
- Soporta tanto sync como async

**Implementación (app/main.py):**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import mlflow
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
import time
from typing import List, Optional

app = FastAPI(
    title="Disease Prediction API",
    description="API para predicción de enfermedades comunes y raras",
    version="2.0.0"
)

# Métricas Prometheus
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
error_counter = Counter('prediction_errors_total', 'Total prediction errors')

# Cargar modelo al inicio
model = mlflow.pyfunc.load_model("models://DiseasePredictor/Production")

class PatientData(BaseModel):
    """Schema de entrada para predicción"""
    patient_id: str = Field(..., description="ID único del paciente (anonimizado)")
    age: int = Field(..., ge=0, le=120, description="Edad del paciente")
    gender: str = Field(..., description="Género del paciente")
    symptoms: List[dict] = Field(..., description="Lista de síntomas")
    comorbidities: Optional[List[str]] = Field(default=[], description="Comorbilidades")
    lab_results: Optional[dict] = Field(default={}, description="Resultados de laboratorio")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['M', 'F', 'Other']:
            raise ValueError('Gender must be M, F, or Other')
        return v
    
    @validator('symptoms')
    def validate_symptoms(cls, v):
        if len(v) == 0:
            raise ValueError('At least one symptom required')
        for symptom in v:
            if 'code' not in symptom or 'severity' not in symptom:
                raise ValueError('Each symptom must have code and severity')
        return v

class PredictionResponse(BaseModel):
    """Schema de respuesta"""
    patient_id: str
    prediction: str
    probability: float
    confidence_interval: tuple
    top_features: List[dict]
    explanation: str
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Realiza predicción para un paciente.
    
    - **patient_id**: ID único del paciente
    - **age**: Edad del paciente
    - **gender**: Género (M/F/Other)
    - **symptoms**: Lista de síntomas con código y severidad
    - **comorbidities**: Lista de comorbilidades (opcional)
    - **lab_results**: Resultados de laboratorio (opcional)
    """
    start_time = time.time()
    
    try:
        # Preprocesar entrada
        features = preprocess_patient_data(patient)
        
        # Predicción
        prediction_counter.inc()
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0, 1]
        
        # Calcular intervalo de confianza
        ci_lower, ci_upper = calculate_confidence_interval(probability)
        
        # Generar explicación con SHAP
        shap_values = calculate_shap(model, features)
        top_features = get_top_features(shap_values)
        explanation = generate_explanation(top_features)
        
        # Registrar latencia
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        # Log para auditoría
        log_prediction(patient.patient_id, prediction, probability)
        
        return PredictionResponse(
            patient_id=patient.patient_id,
            prediction=prediction,
            probability=round(probability, 3),
            confidence_interval=(round(ci_lower, 3), round(ci_upper, 3)),
            top_features=top_features,
            explanation=explanation,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(patients: List[PatientData]):
    """Predicción en batch para múltiples pacientes"""
    results = []
    for patient in patients:
        result = await predict(patient)
        results.append(result)
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "2.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Endpoint para métricas de Prometheus"""
    return generate_latest()

@app.get("/model_info")
async def model_info():
    """Información sobre el modelo en producción"""
    return {
        "model_name": "DiseasePredictor",
        "version": "2.0.0",
        "trained_on": "2024-11-15",
        "performance": {
            "auroc": 0.87,
            "sensitivity": 0.91,
            "specificity": 0.83
        },
        "supported_diseases": ["Common Disease A", "Rare Disease B", ...]
    }

def preprocess_patient_data(patient: PatientData) -> np.ndarray:
    """Preprocesar datos del paciente al formato del modelo"""
    # Implementar feature engineering
    features = []
    # ... lógica de preprocesamiento ...
    return np.array(features).reshape(1, -1)

def calculate_confidence_interval(probability: float) -> tuple:
    """Calcular intervalo de confianza usando bootstrap"""
    # Implementación simplificada
    margin = 0.05
    return (max(0, probability - margin), min(1, probability + margin))

def calculate_shap(model, features):
    """Calcular SHAP values para explicabilidad"""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    return shap_values

def get_top_features(shap_values, n=5):
    """Obtener top N features más importantes"""
    # Implementación...
    return []

def generate_explanation(top_features):
    """Generar explicación en lenguaje natural"""
    # Implementación...
    return "El modelo predice enfermedad basándose en..."

def log_prediction(patient_id, prediction, probability):
    """Log para auditoría y monitoreo"""
    import logging
    logging.info(f"Prediction for {patient_id}: {prediction} (prob: {probability})")
    # También enviar a S3 para análisis posterior
```

**Suposiciones:**
- El API recibirá ~10-100 requests/segundo en promedio
- Los médicos necesitan explicaciones, no solo predicciones
- Las predicciones deben ser auditables (guardadas en logs)

#### 5.4 Opciones de Deployment

##### Opción 1: Cloud (Recomendado para clínicas grandes)

**Tecnologías:**
- **AWS ECS (Elastic Container Service)** o **EKS (Elastic Kubernetes Service)**
- **AWS Application Load Balancer**
- **Amazon API Gateway**: Para rate limiting, autenticación, logging
- **AWS Auto Scaling**: Escalar según demanda

**Arquitectura:**
```
Internet
   ↓
[AWS API Gateway] (autenticación, rate limiting, logging)
   ↓
[Application Load Balancer] (distribución de tráfico)
   ↓
[ECS Cluster / EKS Pods]
   ├── Container 1 (FastAPI + Model)
   ├── Container 2 (FastAPI + Model)
   └── Container N (FastAPI + Model)
```

**Terraform para infrastructure as code:**
```hcl
resource "aws_ecs_cluster" "disease_predictor" {
  name = "disease-predictor-cluster"
}

resource "aws_ecs_task_definition" "predictor_task" {
  family                   = "disease-predictor"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"  # 1 vCPU
  memory                   = "2048"  # 2 GB
  
  container_definitions = jsonencode([{
    name      = "predictor"
    image     = "your-ecr-repo/disease-predictor:latest"
    essential = true
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/disease-predictor"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "predictor_service" {
  name            = "disease-predictor-service"
  cluster         = aws_ecs_cluster.disease_predictor.id
  task_definition = aws_ecs_task_definition.predictor_task.arn
  desired_count   = 3  # 3 instancias para high availability
  
  load_balancer {
    target_group_arn = aws_lb_target_group.predictor_tg.arn
    container_name   = "predictor"
    container_port   = 8000
  }
}
```

**Justificación:**
- ECS/EKS permiten escalar automáticamente según demanda
- Alta disponibilidad (múltiples containers en diferentes availability zones)
- Integración nativa con otros servicios AWS
- Managed service (menos overhead operacional)

**Costo estimado:**
- ECS Fargate: ~$50-150/mes para 3 instancias small
- ALB: ~$20/mes
- API Gateway: $3.50 por millón de requests
- Total: ~$100-200/mes para carga moderada

##### Opción 2: Local (Para clínicas pequeñas/rurales)

**Tecnología:** Docker Desktop + Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  predictor:
    image: disease-predictor:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models
      - ./logs:/logs
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Instrucciones para médicos:**
```bash
# 1. Instalar Docker Desktop (GUI fácil)
# 2. Descargar archivo docker-compose.yml
# 3. Ejecutar en terminal:
docker-compose up -d

# 4. Abrir navegador en http://localhost:8000/docs
# Listo para usar!
```

**Justificación:**
- No requiere conexión a internet
- Datos no salen del computador (privacidad)
- Costo cero de infraestructura
- Fácil de instalar (Docker Desktop tiene GUI)

**Limitaciones:**
- No escala automáticamente
- No tiene alta disponibilidad
- Requiere mantenimiento manual de actualizaciones

##### Opción 3: Edge/Mobile (Para zonas remotas)

**Tecnología:** TensorFlow Lite o ONNX Runtime

**Conversión a TFLite:**
```python
import tensorflow as tf

# Convertir modelo a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Guardar
with open('disease_predictor.tflite', 'wb') as f:
    f.write(tflite_model)
```

**App móvil (React Native + TFLite):**
```javascript
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

// Cargar modelo
const model = await tf.loadLayersModel(
  bundleResourceIO(modelJSON, modelWeights)
);

// Predicción
const prediction = model.predict(tf.tensor2d([patientFeatures]));
```

**Justificación:**
- Funciona sin conexión (ideal para zonas remotas)
- Latencia ultra-baja (<100ms)
- Privacidad total (datos no salen del dispositivo)

**Limitaciones:**
- Modelos limitados en complejidad
- Actualizaciones requieren reinstalar app

**Suposiciones:**
- Los médicos tienen smartphones o tablets
- El modelo es suficientemente pequeño (<50MB)

---

### 6. Predicciones en Producción

#### 6.1 Flujo de Predicción en Tiempo Real

```
Médico ingresa datos del paciente
   ↓
[Frontend Web/Mobile]
   ↓
API Request (HTTPS)
   ↓
[API Gateway] → Autenticación/Rate Limiting
   ↓
[Load Balancer] → Enruta a container disponible
   ↓
[FastAPI Container]
   ├── Validar input (Pydantic)
   ├── Preprocesar features
   ├── Predicción con modelo
   ├── Calcular SHAP (explicabilidad)
   ├── Log para auditoría
   └── Retornar respuesta
   ↓
[Frontend] → Mostrar predicción + explicación
   ↓
Médico toma decisión clínica
```

**SLA objetivo:**
- Latencia p50: <300ms
- Latencia p95: <500ms
- Latencia p99: <1s
- Disponibilidad: 99.9% (permitiendo ~8 horas downtime/año)

#### 6.2 Flujo de Predicción Batch

**Caso de uso:** Screening masivo de pacientes en hospitales

**Tecnología:** Apache Airflow + AWS Batch / SageMaker Batch Transform

**DAG de Airflow:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.batch import BatchOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_disease_screening',
    default_args=default_args,
    description='Daily batch predictions for all patients',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Extract new patient data from data warehouse
    extract_data = PythonOperator(
        task_id='extract_patient_data',
        python_callable=extract_new_patients,
    )
    
    # Task 2: Run batch predictions
    batch_predict = BatchOperator(
        task_id='batch_predictions',
        job_name='disease-prediction-batch',
        job_definition='disease-predictor-job',
        job_queue='batch-processing-queue',
        overrides={
            'vcpus': 4,
            'memory': 8192,
        },
    )
    
    # Task 3: Store results in data warehouse
    store_results = PythonOperator(
        task_id='store_predictions',
        python_callable=store_batch_results,
    )
    
    # Task 4: Generate daily report
    generate_report = PythonOperator(
        task_id='generate_report',
        python_callable=create_daily_report,
    )
    
    # Task 5: Send alerts for high-risk patients
    send_alerts = PythonOperator(
        task_id='send_high_risk_alerts',
        python_callable=alert_high_risk_patients,
    )
    
    # Define dependencies
    extract_data >> batch_predict >> store_results >> [generate_report, send_alerts]
```

**Justificación:**
Airflow es el estándar de facto para orquestación de pipelines de datos. Permite:
- Scheduling automático
- Retry logic
- Monitoreo visual
- Alertas en caso de fallo
- Integración con AWS, GCP, Azure

#### 6.3 Logging y Auditoría

**Objetivo:** Cumplir con regulaciones médicas (trazabilidad completa)

**Qué loggear:**
1. Cada predicción: input, output, timestamp, user_id
2. Latencia de cada request
3. Errores y excepciones
4. Actualizaciones del modelo
5. Accesos al API

**Tecnologías:**
- **AWS CloudWatch Logs**: Logs centralizados
- **AWS S3**: Archivo de logs para análisis histórico
- **Amazon Athena**: Queries SQL sobre logs en S3
- **ELK Stack (alternativa)**: Elasticsearch + Logstash + Kibana

**Implementación:**
```python
import logging
import json
import boto3
from datetime import datetime

# Configurar logger
logger = logging.getLogger(__name__)
s3_client = boto3.client('s3')

def log_prediction(patient_id, input_data, prediction, probability, 
                  latency, user_id, model_version):
    """
    Log detallado de cada predicción para auditoría
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'patient_id': patient_id,
        'user_id': user_id,
        'model_version': model_version,
        'input': input_data,
        'prediction': prediction,
        'probability': float(probability),
        'latency_ms': latency * 1000,
        'environment': 'production'
    }
    
    # Log a CloudWatch
    logger.info(json.dumps(log_entry))
    
    # Guardar en S3 para archivo permanente
    s3_path = f"audit-logs/{datetime.now().strftime('%Y/%m/%d')}/{patient_id}.json"
    s3_client.put_object(
        Bucket='disease-predictor-audit',
        Key=s3_path,
        Body=json.dumps(log_entry),
        ServerSideEncryption='AES256'
    )
```

**Queries de ejemplo con Athena:**
```sql
-- Predicciones por día
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_predictions,
    AVG(probability) as avg_probability,
    AVG(latency_ms) as avg_latency_ms
FROM prediction_logs
WHERE timestamp >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Predicciones de alto riesgo
SELECT *
FROM prediction_logs
WHERE probability > 0.8
  AND prediction = 'Rare Disease X'
  AND timestamp >= CURRENT_DATE - INTERVAL '7' DAY;
```

**Justificación:**
En medicina, la auditoría es legal requirement. Necesitamos poder responder "¿qué predicción se hizo para el paciente X el día Y?" años después.

---

### 7. Monitoreo de Modelos en Producción

#### 7.1 Data Drift Detection

**Problema:** La distribución de los datos de entrada cambia con el tiempo (nuevas cepas de virus, cambios demográficos, etc.)

**Técnicas:**

1. **Population Stability Index (PSI)**:
```python
def calculate_psi(expected, actual, bins=10):
    """
    Calcula PSI entre distribución esperada y actual
    PSI < 0.1: No hay cambio significativo
    0.1 < PSI < 0.2: Cambio moderado
    PSI > 0.2: Cambio significativo (reentrenar!)
    """
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((actual_percents - expected_percents) * 
                  np.log(actual_percents / expected_percents))
    return psi

# Monitorear cada feature
for feature in important_features:
    psi = calculate_psi(
        train_data[feature], 
        production_data_last_week[feature]
    )
    if psi > 0.2:
        alert(f"Significant drift detected in {feature}: PSI = {psi}")
```

2. **KS Test (Kolmogorov-Smirnov)**:
```python
from scipy.stats import ks_2samp

for feature in features:
    statistic, p_value = ks_2samp(
        train_data[feature], 
        production_data[feature]
    )
    if p_value < 0.05:
        alert(f"Distribution change detected in {feature}")
```

**Tecnología:** Evidently AI

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Crear report de drift
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(
    reference_data=train_data,
    current_data=production_data_last_week
)

# Guardar report
drift_report.save_html("reports/drift_report.html")

# Enviar alerta si hay drift
if drift_report.show()['data_drift']['dataset_drift']:
    send_slack_alert("Data drift detected! Check report.")
```

**Justificación:**
Evidently AI es especializada en monitoreo de ML, open-source, y genera reports visuales hermosos que no-técnicos pueden entender.

#### 7.2 Model Performance Monitoring

**Objetivo:** Detectar degradación del modelo antes de que cause problemas

**Métricas a monitorear:**

1. **Online metrics** (disponibles inmediatamente):
   - Distribución de predicciones (¿cambió?)
   - Distribución de probabilidades (¿cambió?)
   - Latencia (¿aumentó?)
   - Tasa de errores

2. **Offline metrics** (requieren ground truth):
   - AUROC, AUPRC (cuando tengamos diagnósticos confirmados)
   - Sensibilidad/Especificidad
   - Calibration error

**Implementación:**
```python
# Dashboard de Grafana con queries a Prometheus

# Query 1: Distribución de predicciones por día
predictions_by_disease{
  prediction_date >= now() - 30d
}

# Query 2: Latencia p95
histogram_quantile(0.95, 
  rate(prediction_latency_seconds_bucket[5m])
)

# Query 3: Tasa de error
rate(prediction_errors_total[5m]) / 
rate(predictions_total[5m])
```

**Alertas automáticas:**
```python
# Alert si AUROC cae por debajo del umbral
if current_auroc < baseline_auroc - 0.05:
    trigger_alert(
        severity='HIGH',
        message=f'Model performance degraded: AUROC {current_auroc:.3f} vs baseline {baseline_auroc:.3f}',
        action='Consider retraining model'
    )

# Alert si latencia excede SLA
if p95_latency > 500:  # ms
    trigger_alert(
        severity='MEDIUM',
        message=f'Latency SLA violated: p95={p95_latency}ms',
        action='Scale up infrastructure or optimize model'
    )
```

**Tecnologías:**
- **Prometheus**: Recolección de métricas (time-series DB)
- **Grafana**: Visualización y alertas
- **PagerDuty/Slack**: Notificaciones

#### 7.3 Fairness Monitoring

**Problema:** El modelo podría tener performance desigual entre subgrupos (género, edad, etnicidad)

**Métricas:**
```python
from aequitas.group import Group
from aequitas.bias import Bias

# Calcular métricas de fairness por subgrupo
g = Group()
xtab, _ = g.get_crosstabs(df, score_thresholds={'score':[0.5]})

b = Bias()
bias_metrics = b.get_disparity_predefined_groups(
    xtab, 
    ref_groups_dict={'gender':'M', 'age':'25-40'},
    alpha=0.05
)

# Visualizar disparidades
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
bias_metrics.plot(kind='bar', ax=ax)
plt.title('Disparidad en métricas por subgrupo')
plt.ylabel('Disparidad (ratio con grupo de referencia)')
plt.axhline(y=1.0, color='r', linestyle='--', label='Paridad')
plt.legend()
plt.tight_layout()
plt.savefig('fairness_metrics.png')
```

**Criterio de aceptación:**
- Diferencia en sensibilidad entre subgrupos < 5%
- Diferencia en especificidad entre subgrupos < 5%

**Justificación:**
En medicina, sesgo algorítmico puede ser literalmente vida o muerte. Múltiples estudios han demostrado que modelos médicos tienen peor performance en minorías.

#### 7.4 Dashboard de Monitoreo

**Tecnología:** Grafana + Prometheus

**Paneles:**

1. **Health Overview:**
   - Availability (% uptime)
   - Request rate (requests/second)
   - Error rate (%)
   - Latency (p50, p95, p99)

2. **Model Performance:**
   - Predicciones por día (por clase)
   - Distribución de probabilidades
   - AUROC/AUPRC (actualizado weekly con ground truth)
   - Calibración

3. **Data Drift:**
   - PSI por feature (top 20 features)
   - Alertas de drift activas

4. **Infrastructure:**
   - CPU usage
   - Memory usage
   - Container count (auto-scaling)
   - Disk usage

**Ejemplo de configuración:**
```yaml
# grafana_dashboard.json
{
  "dashboard": {
    "title": "Disease Prediction Model Monitoring",
    "panels": [
      {
        "title": "Predictions per Day",
        "targets": [{
          "expr": "sum(increase(predictions_total[24h]))"
        }]
      },
      {
        "title": "Model Latency (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "AUROC Trend",
        "targets": [{
          "expr": "model_auroc"
        }]
      }
    ]
  }
}
```

---

### 8. Reentrenamiento y Mejora Continua

#### 8.1 Gatillos de Reentrenamiento

**Reentrenar cuando:**

1. **Drift detectado**: PSI > 0.2 en features críticos
2. **Performance degradation**: AUROC cae >5% respecto a baseline
3. **Nuevos datos**: Acumulación de N casos nuevos (ej. 10,000 para comunes, 50 para raras)
4. **Ventana temporal**: Cada 3 meses automáticamente
5. **Nuevas enfermedades**: Emergencia de nuevas patologías
6. **Feedback médico**: Médicos reportan múltiples predicciones incorrectas

**Implementación:**
```python
def check_retraining_triggers():
    """
    Verifica si es necesario reentrenar el modelo
    """
    triggers = []
    
    # 1. Check drift
    if max_psi > 0.2:
        triggers.append(f"Data drift detected (PSI={max_psi:.3f})")
    
    # 2. Check performance
    if current_auroc < baseline_auroc - 0.05:
        triggers.append(f"Performance degradation (AUROC={current_auroc:.3f})")
    
    # 3. Check new data
    new_samples = count_new_labeled_samples()
    if new_samples > RETRAIN_THRESHOLD:
        triggers.append(f"New data available ({new_samples} samples)")
    
    # 4. Check time
    days_since_training = (datetime.now() - last_training_date).days
    if days_since_training > 90:
        triggers.append(f"Scheduled retrain ({days_since_training} days)")
    
    if triggers:
        initiate_retraining_pipeline(triggers)
    
    return triggers
```

#### 8.2 Automatic Retraining Pipeline

**Tecnología:** AWS SageMaker Pipelines o Airflow

**Pasos:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('model_retraining', schedule_interval='@weekly') as dag:
    
    # 1. Check if retraining needed
    check_triggers = PythonOperator(
        task_id='check_retraining_triggers',
        python_callable=check_retraining_triggers
    )
    
    # 2. Extract new data
    extract_data = PythonOperator(
        task_id='extract_training_data',
        python_callable=extract_latest_data
    )
    
    # 3. Validate data quality
    validate_data = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_schema
    )
    
    # 4. Train new model
    train_model = PythonOperator(
        task_id='train_new_model',
        python_callable=train_and_evaluate
    )
    
    # 5. Compare with production model
    compare_models = PythonOperator(
        task_id='compare_models',
        python_callable=compare_model_performance
    )
    
    # 6. If better, deploy to staging
    deploy_staging = PythonOperator(
        task_id='deploy_to_staging',
        python_callable=deploy_to_staging_env
    )
    
    # 7. Run A/B test
    ab_test = PythonOperator(
        task_id='run_ab_test',
        python_callable=run_ab_test,
        execution_timeout=timedelta(days=7)
    )
    
    # 8. If successful, promote to production
    promote_prod = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_to_prod
    )
    
    check_triggers >> extract_data >> validate_data >> train_model >> \
    compare_models >> deploy_staging >> ab_test >> promote_prod
```

**Justificación:**
Automatizar el reentrenamiento reduce el tiempo de respuesta a cambios y asegura que el modelo siempre esté actualizado.

#### 8.3 Active Learning para Enfermedades Raras

**Problema:** Para enfermedades raras, cada caso nuevo es valioso. ¿Cómo priorizar qué casos etiquetar primero?

**Estrategia:**

1. **Uncertainty Sampling**: Etiquetar casos donde el modelo está más incierto
2. **Diversity Sampling**: Etiquetar casos diferentes a los ya vistos
3. **Error Analysis**: Priorizar casos similares a errores pasados

**Implementación:**
```python
def select_cases_for_labeling(unlabeled_data, model, n_samples=50):
    """
    Selecciona los casos más informativos para etiquetar
    usando Active Learning
    """
    # 1. Predicciones en datos sin etiquetar
    probas = model.predict_proba(unlabeled_data)
    
    # 2. Uncertainty score (entropía)
    from scipy.stats import entropy
    uncertainties = entropy(probas, axis=1)
    
    # 3. Diversity score (distancia a casos ya vistos)
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(unlabeled_data, labeled_data)
    diversity_scores = distances.min(axis=1)
    
    # 4. Score combinado
    combined_score = 0.6 * uncertainties + 0.4 * diversity_scores
    
    # 5. Seleccionar top N
    top_indices = np.argsort(combined_score)[-n_samples:]
    
    return unlabeled_data[top_indices]

# Enviar a médicos para etiquetar
prioritized_cases = select_cases_for_labeling(new_patients, model)
send_to_annotation_queue(prioritized_cases)
```

**Justificación:**
Active learning puede reducir la cantidad de etiquetado necesario en 50-70%, crítico cuando el etiquetado (diagnóstico médico) es costoso y lento.

#### 8.4 Human-in-the-Loop

**Implementación:** Streamlit app para feedback de médicos

```python
import streamlit as st

st.title('Medical Feedback Interface')

# Mostrar predicción del modelo
st.subheader('Model Prediction')
st.write(f"Predicted Disease: {prediction}")
st.write(f"Confidence: {probability:.2%}")
st.write(f"Explanation: {explanation}")

# Solicitar feedback del médico
st.subheader('Your Diagnosis')
actual_diagnosis = st.selectbox(
    'Confirmed diagnosis:',
    options=disease_list + ['Model is correct', 'Uncertain']
)

if actual_diagnosis != 'Uncertain':
    confidence = st.slider('Confidence in your diagnosis:', 0, 100, 80)
    notes = st.text_area('Clinical notes (optional):')
    
    if st.button('Submit Feedback'):
        save_feedback(patient_id, prediction, actual_diagnosis, 
                     confidence, notes)
        st.success('Thank you! Feedback saved.')
        
        # Si hay discrepancia, agregar a casos para revisión
        if actual_diagnosis != prediction and actual_diagnosis != 'Model is correct':
            flag_for_review(patient_id, prediction, actual_diagnosis)
```

**Justificación:**
El feedback médico es la "ground truth" más valiosa. Facilitar su recolección asegura mejora continua del modelo.

---

### 9. Consideraciones Adicionales

#### 9.1 Seguridad y Privacidad

**Medidas:**

1. **Encriptación:**
   - En tránsito: TLS 1.3 para todas las comunicaciones
   - En reposo: AES-256 para datos en S3/RDS
   - PII/PHI: Encriptación a nivel de aplicación con KMS

2. **Autenticación y Autorización:**
   - OAuth 2.0 / OpenID Connect
   - Role-Based Access Control (RBAC)
   - MFA para usuarios administrativos

3. **Anonimización:**
   - Hash irreversible de IDs de pacientes
   - Eliminación de nombres, direcciones, teléfonos
   - Generalización de datos (edad exacta → rangos)

4. **Auditoría:**
   - Log de todos los accesos a datos
   - Retention de logs por 7 años (compliance)
   - Regular security audits

**Tecnologías:**
- **AWS KMS**: Key Management Service
- **AWS Secrets Manager**: Para API keys y credenciales
- **HashiCorp Vault (alternativa)**: Secret management
- **OAuth Provider**: AWS Cognito o Auth0

**Compliance:**
- HIPAA (USA)
- GDPR (Europa)
- Ley 1581/2012 (Colombia)

#### 9.2 Disaster Recovery

**Estrategia:**

1. **Backups:**
   - Modelos: Versionados en S3 con cross-region replication
   - Datos: Snapshots diarios de RDS/Snowflake
   - Código: Git (inherently backed up)

2. **Multi-Region Deployment:**
   - Primary: us-east-1
   - Secondary: us-west-2
   - Failover automático con Route 53

3. **RTO/RPO Targets:**
   - RTO (Recovery Time Objective): 4 horas
   - RPO (Recovery Point Objective): 24 horas

**Justificación:**
Para sistemas médicos, alta disponibilidad es crítica. Un downtime prolongado podría impedir diagnósticos urgentes.

#### 9.3 Costos

**Estimación mensual (carga moderada: 100K predicciones/mes):**

| Componente | Costo (USD/mes) |
|-----------|----------------|
| ECS Fargate (3 containers) | $100 |
| Application Load Balancer | $20 |
| API Gateway | $10 |
| S3 Storage (1 TB) | $23 |
| Snowflake (Small warehouse) | $80 |
| RDS PostgreSQL (db.t3.medium) | $60 |
| CloudWatch Logs | $15 |
| SageMaker (training 10h/mes) | $50 |
| **TOTAL** | **~$358/mes** |

**Optimizaciones posibles:**
- Usar Spot Instances para training: -70% costo
- Comprimir logs antiguos a Glacier: -80% storage cost
- Reserved Instances si uso es constante: -30% compute cost

---

## Suposiciones Globales

1. **Datos:**
   - Disponibilidad de al menos 50,000 casos de enfermedades comunes
   - Al menos 5-10 casos por enfermedad huérfana
   - Datos históricos de los últimos 3-5 años
   - Calidad de datos razonable (>80% completitud)

2. **Infraestructura:**
   - Acceso a cloud provider (AWS preferentemente)
   - Budget de ~$500-1000/mes para infraestructura
   - Equipo técnico con experiencia en Python y Docker

3. **Stakeholders:**
   - Médicos dispuestos a proporcionar feedback
   - Comité de ética para revisar decisiones algorítmicas
   - Administradores que entienden el valor de ML

4. **Regulatorio:**
   - Cumplimiento HIPAA/GDPR es mandatorio
   - Modelos deben ser auditables
   - Explicabilidad es requerida por reguladores

5. **Performance:**
   - Sensibilidad >90% es más importante que precisión
   - Latencia <500ms es aceptable para uso clínico
   - Disponibilidad de 99.9% es suficiente (no 99.99%)

---

## Próximos Pasos

1. **Fase 1 (Mes 1-2):**
   - Setup de infraestructura base (S3, Snowflake, GitHub)
   - Ingesta y limpieza de datos históricos
   - EDA inicial y feature engineering

2. **Fase 2 (Mes 3-4):**
   - Entrenamiento de modelos baseline
   - Implementación de estrategias para enfermedades raras
   - Evaluación y selección de modelos

3. **Fase 3 (Mes 5):**
   - Containerización y deployment a staging
   - Validación clínica con médicos
   - Ajustes basados en feedback

4. **Fase 4 (Mes 6):**
   - Deployment a producción
   - Setup de monitoreo y alertas
   - Documentación final

5. **Post-Launch:**
   - Recolección de feedback continuo
   - Reentrenamientos periódicos
   - Expansión a más enfermedades

---
