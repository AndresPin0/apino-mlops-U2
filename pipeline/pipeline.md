# Pipeline de MLOps – Predicción de estados de enfermedad (comunes y huérfanas)

**Objetivo:** dado un vector de síntomas/variables clínicas, predecir el estado: **NO ENFERMO | ENFERMEDAD LEVE | ENFERMEDAD AGUDA | ENFERMEDAD CRÓNICA**, contemplando patologías **comunes** (muchos datos) y **huérfanas** (pocos datos).

---

## 1. Diseño (restricciones y tipos de datos)

- **Privacidad/Gobernanza:** cumplimiento regulatorio (p. ej., HIPAA/GDPR/normativa local), control de PHI/PII, auditoría y trazabilidad.
- **Sesgos y desbalance:** prevalencias muy bajas (huérfanas), variabilidad entre hospitales; riesgo de *data leakage* por paciente/tiempo.
- **Tipos de datos:**
  - Estructurados: demografía, signos vitales, laboratorio, comorbilidades, tratamientos.
  - Texto clínico: notas médicas/síntomas (se requiere anonimización).
  - Señales/Imágenes (opcional, fuera del demo pero previsto en el diseño).
- **Salida del modelo:** probabilidad/etiqueta discreta + **incertidumbre** + **explicabilidad** para uso clínico.

---

## 2. Desarrollo (fuentes, manejo, modelos, validación)

### 2.1 Fuentes e ingesta
- EMR/HIS, LIS, PACS, formularios de admisión/triage.
- **Catálogo y linaje de datos**; contratos de intercambio; control de acceso.

### 2.2 Calidad, anonimización y *feature store*
- Reglas de validación clínica (rangos y unidades), detección/corrección de atípicos, imputación prudente (KNN, *carry-forward* controlado).
- Normalización terminológica (p. ej., mapeos a catálogos internos).
- **Feature Store** con *schemas* versionados (p. ej., validación tipo pydantic) y particionamiento por paciente/episodio.

### 2.3 Estrategia de modelado
- **Comunes (muchos datos):** modelos discriminativos (Logistic/GBM/XGBoost/NN tabulares), **calibración** (Platt/Isotónica), explicabilidad (SHAP), *cost-sensitive learning*.
- **Huérfanas (pocos datos):**
  - **Transfer learning / Multi-task** desde tareas/comunes.
  - **Few-shot / Meta-learning** (p. ej., prototípicos sobre *embeddings* tabulares/texto).
  - **Bayesianos** para incertidumbre; **anomaly / one-class** para *out-of-distribution*.
  - **Weak supervision / síntesis tabular** (SMOTE con cautela clínica y validación).
- **Ingeniería de decisiones clínicas:** umbrales por contexto (sensibilidad fija), *decision-curve analysis* y costos.

### 2.4 Validación y test
- **Partición por paciente** y **cronológica** (train < val < test) para evitar *leakage*.
- Métricas: **AUPRC** (desbalance), AUROC, sensibilidad/especificidad objetivo, **ECE** (calibración), además de métricas por subgrupos (fairness).
- Robustez: *stress tests* (faltantes/ruido), OOD.

---

## 3. Producción (despliegue, monitoreo, reentrenos)

- **Despliegue:** servicio API contenedorizado (Docker/K8s), *model registry*, control de versiones y *inference schema* estricto.
- **Monitoreo:** *data drift* (PSI), *prediction drift*, desempeño con verdad terreno retardada, alarmas; trazabilidad por predicción.
- **Mejora continua:** *human-in-the-loop*, **active learning** para priorizar anotación de casos raros/incertidumbre alta, **gatillos de reentrenamiento** (drift + N casos + ventana de tiempo).
- **Documentación y riesgo:** *model cards*, comité clínico-técnico, auditorías.

---


## 4. Conexión con el servicio del punto 2
- El servicio expone el **inference schema** (tres entradas mínimas para el demo) y devuelve una de las 4 clases.
- El monitoreo registra la distribución de entradas y salidas para detectar *drift*.
- Cuando existan datos reales y etiquetas clínicas, la función de reglas puede **sustituirse por un modelo entrenado** manteniendo la misma interfaz.
