# CHANGELOG - Reestructuración del Pipeline MLOps

## Comparación: Propuesta Original (V1.0) vs Propuesta Reestructurada (V2.0)

---

## Resumen 

Este documento detalla los cambios realizados entre la **propuesta inicial** del pipeline MLOps (Semana 1) y la **propuesta reestructurada** (Unidad 2), incorporando todo el conocimiento adquirido durante el curso.

**Fecha de propuesta original:** Semana 1 del curso  
**Fecha de reestructuración:** Noviembre 2024  
**Tipo de cambio:** Reestructuración mayor con expansión significativa

---

## Cambios Principales

### 1. **ESTRUCTURA Y ORGANIZACIÓN**

#### V1.0 (Original):
- Estructura conceptual de alto nivel
- 4 secciones principales: Diseño, Desarrollo, Producción, Conexión con servicio
- Aproximadamente 60 líneas de documentación
- Enfoque genérico sin tecnologías específicas

#### V2.0 (Reestructurada):
- Estructura detallada end-to-end con 9 secciones principales
- Más de 1000 líneas de documentación técnica
- Cada etapa con subsecciones detalladas
- Diagramas ASCII completos del sistema
- Ejemplos de código implementables

**Impacto:** La propuesta es ahora un documento implementable, no solo conceptual.

---

### 2. **TECNOLOGÍAS ESPECÍFICAS**

#### V1.0 (Original):
- **Mencionadas genéricamente:** Docker, Kubernetes, feature stores, schemas versionados
- **Sin especificar:** Proveedores cloud, herramientas de monitoreo, frameworks
- **Sin justificación:** No se explicaba por qué usar cada tecnología

#### V2.0 (Reestructurada):

| Categoría | Tecnologías Específicas | Justificación |
|-----------|------------------------|---------------|
| **Control de versiones** | GitHub + GitHub Actions | CI/CD integrado, estándar de industria |
| **Storage** | AWS S3, Snowflake | Escalabilidad, compliance HIPAA |
| **Versionado de datos** | DVC | Trazabilidad de datasets grandes |
| **Notebooks** | Jupyter + EC2 | Estándar para DS, recursos escalables |
| **Frameworks ML** | XGBoost, PyTorch, scikit-learn | Específicos según tipo de datos |
| **Tracking** | MLflow | Versionamiento de modelos y experimentos |
| **Tuning** | Optuna | Optimización automática de hiperparámetros |
| **API Framework** | FastAPI | Performance + validación automática |
| **Containerización** | Docker + Docker Compose | Reproducibilidad garantizada |
| **Orchestration** | Apache Airflow | Estándar para pipelines de datos |
| **Deployment Cloud** | AWS ECS/EKS | Escalabilidad automática |
| **Deployment Local** | Docker Desktop | Para clínicas con recursos limitados |
| **Edge** | TensorFlow Lite | Para zonas sin conexión |
| **Monitoring** | Prometheus + Grafana + Evidently AI | Métricas técnicas + drift detection |
| **Explicabilidad** | SHAP + LIME | Crítico para uso médico |
| **Visualización** | Streamlit | Dashboards para stakeholders |
| **Logging** | CloudWatch + S3 + Athena | Auditoría completa |
| **Seguridad** | AWS KMS, Secrets Manager | Compliance |

**Impacto:** Cada tecnología está justificada y es implementable inmediatamente.

---

### 3. **INGESTA Y ALMACENAMIENTO DE DATOS**

#### V1.0 (Original):
```
- Fuentes: EMR/HIS, LIS, PACS, formularios
- Catálogo y linaje de datos
- Control de acceso
```
(3 líneas, sin detalles)

#### V2.0 (Reestructurada):
- **Fuentes detalladas:** EMR/HIS (formatos HL7/FHIR), LIS, PACS, APIs externas, datos genómicos
- **Arquitectura de storage:**
  - Raw data en S3 con estructura específica de buckets
  - Lifecycle policies (mover a Glacier después de 1 año)
  - Encriptación SSE-KMS
  - Access logs habilitados
- **Preprocesamiento:**
  - Validación con Pydantic/Pandera
  - Anonimización de PII/PHI con hashing
  - Normalización de unidades y códigos
  - Versionado con DVC
- **Data Warehouse:**
  - Snowflake (preferido) vs Redshift (alternativa)
  - Esquemas de tablas SQL específicos (patients, symptoms, diagnoses)
- **Código implementable:** Scripts de ejemplo para cada paso

**Impacto:** De descripción conceptual a arquitectura implementable con código.

---

### 4. **MANEJO DE ENFERMEDADES RARAS (MEJORA CRÍTICA)**

#### V1.0 (Original):
```
- Transfer learning / Multi-task desde tareas comunes
- Few-shot / Meta-learning
- Bayesianos para incertidumbre
- Anomaly / one-class detection
- Weak supervision / síntesis tabular (SMOTE con cautela)
```
(Conceptos mencionados sin implementación)

#### V2.0 (Reestructurada):

**Estrategias expandidas con implementación:**

1. **Few-Shot Learning:**
   - Código completo de Siamese Networks
   - Arquitectura específica con PyTorch
   - Función de distancia implementada

2. **Transfer Learning:**
   - Pipeline completo: pre-train → fine-tune
   - Learning rates específicos
   - Estrategia de congelamiento de capas

3. **Ensemble Híbrido:**
   - Combinación de ML + reglas médicas
   - Pesos configurables (0.7 ML + 0.3 reglas)
   - Integración con knowledge bases médicas

4. **Active Learning:**
   - Algoritmo de selección de casos
   - Uncertainty sampling + Diversity sampling
   - Sistema de priorización para etiquetado médico
   - Código Python completo

**Impacto:** Las estrategias ahora son implementables, no solo teóricas.

---

### 5. **ENTRENAMIENTO Y EXPERIMENTACIÓN**

#### V1.0 (Original):
```
- Modelos discriminativos (Logistic/GBM/XGBoost/NN tabulares)
- Calibración (Platt/Isotónica)
- Explicabilidad (SHAP)
- Cost-sensitive learning
```
(Conceptos sin detalles de implementación)

#### V2.0 (Reestructurada):

**Añadido:**
- **Feature Engineering detallado:**
  - 5 tipos de features: agregación, temporal, interacción, conocimiento médico, texto
  - Código con Featuretools para automated feature engineering
  - Estrategias específicas para datos médicos

- **Hyperparameter Tuning:**
  - Código completo con Optuna
  - Configuración de búsqueda para XGBoost
  - 100+ trials automáticos

- **Experiment Tracking:**
  - Integración completa con MLflow
  - Log de parámetros, métricas, modelos
  - Código de ejemplo para tracking

- **Manejo de Desbalance:**
  - Estrategias diferenciadas para comunes vs raras
  - Implementación de SMOTE
  - Class weighting específico
  - Código completo

- **Calibración:**
  - Implementación con scikit-learn
  - Visualización de calibration curves
  - Métricas (ECE, Brier Score)

**Impacto:** De conceptos a código ejecutable con ejemplos reales.

---

### 6. **EVALUACIÓN Y SELECCIÓN DE MODELOS**

#### V1.0 (Original):
```
- Métricas: AUPRC, AUROC, sensibilidad/especificidad, ECE
- Partición por paciente y cronológica
- Robustez: stress tests, OOD
```
(Sin detalles de implementación)

#### V2.0 (Reestructurada):

**Expansión significativa:**

1. **Métricas detalladas:**
   - Código para calcular todas las métricas
   - Métricas específicas para enfermedades raras
   - Optimización de umbrales para sensibilidad objetivo (95%)

2. **Validación cruzada:**
   - Stratified K-Fold implementado
   - Validación temporal (train 2020-2022, val 2023 Q1-Q3, test 2023 Q4)
   - Código completo

3. **Explicabilidad:**
   - SHAP: código completo con visualizaciones
   - LIME: implementación alternativa
   - Feature importance para tree models
   - Explicaciones para médicos (no-técnicos)

4. **Dashboards para stakeholders:**
   - Código completo de Streamlit
   - Visualizaciones con Plotly
   - Confusion matrix, ROC curves, feature importance
   - Métricas principales en cards

5. **Criterios de aceptación específicos:**
   - AUROC ≥ 0.85 (comunes), ≥ 0.75 (raras)
   - Sensibilidad ≥ 90% (enfermedades graves)
   - ECE ≤ 0.1
   - Latencia ≤ 500ms
   - Throughput ≥ 100 pred/s
   - Modelo ≤ 500MB

**Impacto:** Criterios cuantificables y código para evaluación completa.

---

### 7. **DEPLOYMENT (MEJORA MAYOR)**

#### V1.0 (Original):
```
- Servicio API contenedorizado (Docker/K8s)
- Model registry
- Control de versiones
- Inference schema estricto
```
(Conceptual, sin detalles)

#### V2.0 (Reestructurada):

**Expansión completa:**

1. **Model Registry:**
   - MLflow Model Registry implementado
   - Staging workflow (dev → staging → production)
   - Código completo de transición de versiones

2. **Containerización:**
   - Dockerfile completo y optimizado
   - Multi-stage builds
   - Health checks implementados
   - requirements.txt específico

3. **API de Predicción:**
   - FastAPI completo (200+ líneas de código)
   - Validación automática con Pydantic
   - Endpoints: /predict, /predict_batch, /health, /metrics, /model_info
   - Métricas de Prometheus integradas
   - Manejo de errores robusto
   - Logging para auditoría

4. **Opciones de Deployment:**

   **a) Cloud (Nuevo):**
   - Arquitectura completa: API Gateway → ALB → ECS/EKS
   - Terraform code para infrastructure as code
   - Auto-scaling configurado
   - Multi-AZ para high availability
   - Estimación de costos: $100-200/mes

   **b) Local (Nuevo):**
   - Docker Compose completo
   - Instrucciones paso a paso para médicos
   - Sin dependencia de internet
   - Costo cero

   **c) Edge/Mobile (Nuevo):**
   - Conversión a TensorFlow Lite
   - App React Native con TFLite
   - Para zonas sin conexión
   - Latencia <100ms

**Impacto:** De concepto a 3 opciones de deployment completamente implementables.

---

### 8. **PREDICCIONES EN PRODUCCIÓN**

#### V1.0 (Original):
- No especificado en la propuesta original

#### V2.0 (Reestructurada):

**Añadido completamente:**

1. **Flujo de Predicción Real-Time:**
   - Diagrama de flujo completo
   - SLAs específicos (p50<300ms, p95<500ms, p99<1s)
   - Disponibilidad 99.9%

2. **Predicción Batch:**
   - DAG de Airflow completo (100+ líneas)
   - 5 tareas: extract → predict → store → report → alert
   - Schedule diario a las 2 AM
   - Manejo de errores y retries

3. **Logging y Auditoría:**
   - Función completa de logging
   - Guardado en CloudWatch + S3
   - Queries SQL con Athena
   - Compliance con regulaciones (7 años de retención)

**Impacto:** Sistema de producción completo, no mencionado en V1.0.

---

### 9. **MONITOREO (MEJORA CRÍTICA)**

#### V1.0 (Original):
```
- Data drift (PSI)
- Prediction drift
- Desempeño con verdad terreno retardada
- Alarmas
- Trazabilidad por predicción
```
(Conceptos sin implementación)

#### V2.0 (Reestructurada):

**Expansión completa:**

1. **Data Drift Detection:**
   - **PSI (Population Stability Index):** Código completo con interpretación
   - **KS Test:** Implementación estadística
   - **Evidently AI:** Integración completa con reports HTML
   - Alertas automáticas cuando PSI > 0.2

2. **Model Performance Monitoring:**
   - Métricas online (disponibles inmediatamente): distribución de predicciones, latencia, tasa de errores
   - Métricas offline (requieren ground truth): AUROC, calibración
   - Queries de Prometheus/Grafana

3. **Fairness Monitoring (Nuevo):**
   - Detección de sesgo por subgrupos (género, edad, etnicidad)
   - Código con Aequitas library
   - Criterio: diferencia <5% entre subgrupos
   - Crítico para ética médica

4. **Dashboard de Monitoreo:**
   - 4 paneles: Health, Model Performance, Data Drift, Infrastructure
   - Configuración completa de Grafana
   - Alertas a Slack/PagerDuty

**Impacto:** De conceptos a sistema de monitoreo completo y ético.

---

### 10. **REENTRENAMIENTO Y MEJORA CONTINUA (NUEVO)**

#### V1.0 (Original):
```
- Human-in-the-loop
- Active learning para priorizar anotación
- Gatillos de reentrenamiento (drift + N casos + ventana de tiempo)
```
(Conceptos sin detalles)

#### V2.0 (Reestructurada):

**Expansión completa:**

1. **Gatillos de Reentrenamiento:**
   - 6 triggers específicos: drift (PSI>0.2), performance degradation (>5%), nuevos datos, temporal (cada 3 meses), nuevas enfermedades, feedback médico
   - Función Python completa para detección

2. **Automatic Retraining Pipeline:**
   - DAG de Airflow con 8 pasos completos
   - Check triggers → Extract → Validate → Train → Compare → Deploy staging → A/B test → Promote prod
   - Código implementable

3. **Active Learning:**
   - Algoritmo completo (uncertainty + diversity sampling)
   - Reducción de 50-70% en necesidad de etiquetado
   - Cola de priorización para médicos

4. **Human-in-the-Loop:**
   - Interfaz Streamlit completa
   - Feedback médico estructurado
   - Ground truth collection
   - Flagging automático de discrepancias

**Impacto:** Sistema de mejora continua completamente automatizado.

---

### 11. **SEGURIDAD Y COMPLIANCE (NUEVO)**

#### V1.0 (Original):
```
- Cumplimiento regulatorio (HIPAA/GDPR)
- Control de PHI/PII
- Auditoría y trazabilidad
```
(Mencionado brevemente)

#### V2.0 (Reestructurada):

**Expansión mayor:**

1. **Encriptación:**
   - En tránsito: TLS 1.3
   - En reposo: AES-256
   - PII/PHI: KMS

2. **Autenticación:**
   - OAuth 2.0 / OpenID Connect
   - RBAC (Role-Based Access Control)
   - MFA para admins

3. **Anonimización:**
   - Técnicas específicas (hashing, generalización)
   - Código de ejemplo

4. **Compliance:**
   - HIPAA (USA)
   - GDPR (Europa)
   - Ley 1581/2012 (Colombia)

5. **Disaster Recovery (Nuevo):**
   - Backups: modelos, datos, código
   - Multi-region deployment
   - RTO: 4 horas, RPO: 24 horas

**Impacto:** De mención breve a plan de seguridad completo.

---

### 12. **COSTOS Y RECURSOS (NUEVO)**

#### V1.0 (Original):
- No especificado

#### V2.0 (Reestructurada):

**Añadido:**
- Estimación detallada por componente
- Total: ~$358/mes para carga moderada
- Estrategias de optimización:
  - Spot Instances: -70%
  - Reserved Instances: -30%
  - Glacier para logs: -80% storage

**Impacto:** Visibilidad completa de inversión necesaria.

---

### 13. **SUPOSICIONES (NUEVO)**

#### V1.0 (Original):
- Suposiciones mezcladas en el texto

#### V2.0 (Reestructurada):

**Añadido:**
- Sección completa de "Suposiciones Globales"
- 5 categorías: Datos, Infraestructura, Stakeholders, Regulatorio, Performance
- Específicas y cuantificables

**Impacto:** Claridad total sobre prerequisites del proyecto.

---

### 14. **DIAGRAMAS Y VISUALIZACIÓN**

#### V1.0 (Original):
- Sin diagramas

#### V2.0 (Reestructurada):

**Añadido:**
- Diagrama ASCII completo del pipeline (100+ líneas)
- Diagrama de arquitectura general
- Diagramas de flujo (real-time y batch)
- Arquitectura de deployment

**Impacto:** Visualización completa del sistema.

---

### 15. **DOCUMENTACIÓN Y REFERENCIAS (NUEVO)**

#### V1.0 (Original):
- Sin referencias

#### V2.0 (Reestructurada):

**Añadido:**
- Plan de implementación por fases (6 meses)
- Referencias a papers académicos
- Links a herramientas
- Guías de compliance
- Recursos educativos

**Impacate:** Propuesta lista para ejecutar con timeline y recursos.

---

## Comparación Cuantitativa

| Métrica | V1.0 (Original) | V2.0 (Reestructurada) | Cambio |
|---------|----------------|----------------------|---------|
| **Líneas de documentación** | ~60 | ~1,900 | +3,067% |
| **Secciones principales** | 4 | 9 | +125% |
| **Tecnologías específicas mencionadas** | ~5 | ~30 | +500% |
| **Ejemplos de código** | 0 | 25+ | ∞ |
| **Diagramas** | 0 | 4 | ∞ |
| **Suposiciones explícitas** | ~5 | 20+ | +300% |
| **Opciones de deployment** | 1 (genérica) | 3 (detalladas) | +200% |
| **Criterios de aceptación cuantificables** | 0 | 7 | ∞ |

---

## Cambios Conceptuales Clave

### 1. **De Genérico a Específico**
- V1.0: "Feature stores con schemas versionados"
- V2.0: "Feature engineering con Featuretools + validación con Pydantic + versionado con DVC + almacenamiento en Snowflake con esquema SQL específico"

### 2. **De Conceptual a Implementable**
- V1.0: "Transfer learning para enfermedades huérfanas"
- V2.0: Código completo de 50+ líneas implementando Siamese Networks + estrategia de fine-tuning + ensemble híbrido

### 3. **De Descriptivo a Cuantificable**
- V1.0: "Monitoreo de drift"
- V2.0: "PSI > 0.2 dispara alerta, reentrenamiento automático si persiste por 2 semanas"

### 4. **De Teórico a Práctico**
- V1.0: Enfoque académico
- V2.0: Enfoque de ingeniería con costos, timelines, y trade-offs

---

## Decisiones de Diseño Clave Añadidas en V2.0

1. **Múltiples opciones de deployment** (cloud, local, edge) para diferentes contextos clínicos
2. **Active learning** para optimizar el etiquetado de enfermedades raras
3. **Fairness monitoring** para detectar sesgos éticos
4. **Disaster recovery** para garantizar continuidad del servicio
5. **A/B testing** antes de promover modelos a producción
6. **Human-in-the-loop** con interfaz dedicada para médicos
7. **Estimación de costos** para planificación presupuestaria
8. **Plan de implementación por fases** para ejecución realista

---

## Elementos Mantenidos de V1.0

**Conceptos core que se mantuvieron (pero se expandieron):**

1. Enfoque dual para enfermedades comunes vs raras
2. Importancia de explicabilidad
3. Validación clínica
4. Privacidad y compliance
5. Monitoreo de drift
6. Mejora continua

**Cambio:** Estos conceptos ahora tienen implementación detallada.

---

## Justificación de los Cambios

### ¿Por qué esta expansión masiva?

1. **Aprendizajes del curso:** Cada unidad del curso aportó conocimientos específicos que se integraron:
   - Unidad 1: CI/CD y versionamiento
   - Unidad 2: Deployment y contenedores
   - Unidad 3: Monitoreo y observabilidad
   - Unidad 4: MLOps best practices

2. **Viabilidad de implementación:** V1.0 era conceptual, V2.0 es un plan de implementación ejecutable

3. **Realismo operacional:** Se agregaron consideraciones prácticas (costos, seguridad, disaster recovery) que cualquier proyecto real necesita

4. **Responsabilidad ética:** Se expandió significativamente la sección de fairness y compliance, crítico en medicina

5. **Experiencia de usuario:** Se añadieron múltiples opciones de deployment para diferentes contextos (hospital grande vs clínica rural)

---

## Próximos Pasos Recomendados

1. **Validar con stakeholders:**
   - Médicos: revisar usabilidad y explicabilidad
   - Administradores: revisar costos y timeline
   - Legal: revisar compliance

2. **Prototipo rápido:**
   - Implementar versión simplificada en 2 semanas
   - Probar con dataset público (MIMIC-III)

3. **Infraestructura:**
   - Setup AWS account
   - Configurar GitHub repo con CI/CD
   - Provisionar Snowflake/Redshift

4. **Equipo:**
   - 2 ML Engineers
   - 1 Data Engineer
   - 1 DevOps
   - 1 Medical domain expert

---
