# Detección de empresas fantasmas en redes de facturación electrónica

Código del trabajo de titulación de Ingeniería en Sistemas de Información (Universidad Central del
Ecuador). El repositorio implementa un análisis comparativo entre Machine Learning tradicional,
analítica de grafos y Redes Neuronales de Grafos (GNN) para detectar empresas fantasmas en una red
sintética de facturación electrónica.

## Objetivo

Comparar en igualdad de condiciones tres enfoques de detección de empresas fantasmas: modelos tabulares
con variables transaccionales, los mismos modelos enriquecidos con métricas estructurales de la red y
GNN que aprenden directamente de la estructura del grafo de facturación.

## Grupos experimentales

| Grupo | Modelos | Información usada |
|-------|---------|-------------------|
| G1 | Random Forest, XGBoost | Variables transaccionales por empresa |
| G2 | Random Forest, XGBoost | Variables transaccionales + métricas estructurales (grado, PageRank, clustering, comunidad) |
| G3 | GraphSAGE, GAT | Grafo de facturación con variables de nodo |

Los tres grupos se entrenan y evalúan sobre exactamente las mismas particiones de empresas.

**Métrica principal:** AUC-PR (average precision).
**Métricas complementarias:** F1, precisión, exhaustividad (recall), tasa de falsos positivos (FPR),
precision@k y AUC-ROC.

## Estructura

```
configs/              parámetros de dataset, modelos y experimentos (YAML)
data/                 datos generados (excluido de git)
src/efd/
  data.py             ExperimentData: vista tabular y vista de grafo alineadas por nodo
  registry.py         registro genérico de componentes por nombre
  reproducibility.py  control de semillas
  generation/         generador de la red sintética de facturación
  graph/              construcción del grafo y métricas estructurales
  features/           variables tabulares (G1) y estructurales (G2)
  models/             interfaz BaseDetector, registro e implementaciones
  evaluation/         particiones, métricas y política de umbral
  explainability/     SHAP (G1/G2) y GNNExplainer (G3)
  experiments/        orquestador y registro de ejecuciones
results/runs/         una carpeta por ejecución (excluida de git)
tests/                pruebas con pytest
docs/architecture.md  decisiones de diseño y su justificación
```

## Instalación

Requiere Python 3.11 o superior.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

## Uso

Ejecutar las pruebas:

```bash
pytest
```

Ejecutar un experimento a partir de una configuración:

```bash
python -m efd.experiments.runner --config configs/example.yaml
```

Cada ejecución se guarda en `results/runs/<AAAAMMDD-HHMMSS>_<experimento>/` con la configuración usada,
la semilla, las particiones, las métricas en JSON y las versiones de las librerías.

`configs/example.yaml` documenta la forma de una configuración: semilla, parámetros del generador,
particiones, política de umbral, métricas y, para cada grupo, sus variables, modelos e explicador. Los
nombres de componentes que aparecen en el archivo son los registrados en el código.

### Estado de la implementación

Implementado y probado: `ExperimentData`, `BaseDetector`, `BaseExplainer`, los registros por nombre, las
métricas, las particiones y la política de umbral. El generador, la construcción del grafo, los
constructores de variables, los modelos y explicadores concretos y el orquestador tienen definidas sus
interfaces y lanzan `NotImplementedError`; por eso el comando de ejecución de experimentos todavía no
produce resultados.

## Arquitectura

- **Un único contenedor de datos.** `ExperimentData` reúne la vista tabular y la vista de grafo de la
  misma población de empresas, alineadas por posición de nodo.
- **Particiones compartidas.** `evaluation/splits.py` genera un k-fold estratificado sobre nodos, con
  validación interna, a partir solo de las etiquetas y la semilla. Todos los grupos usan esos índices.
- **Interfaz común de detectores.** `BaseDetector` define `fit(data, train_idx, val_idx)` y
  `predict_proba(data, idx)`; los tabulares leen la vista tabular y los de grafo la vista de grafo.
- **Umbral elegido en validación.** Por defecto, el que maximiza F1; se aplica sin cambios en prueba.
- **Registro por nombre.** Modelos, métricas, políticas de umbral, variables y explicadores se
  seleccionan desde el YAML; agregar uno no exige modificar el orquestador ni el evaluador.
- **Una carpeta por ejecución.** Configuración, semilla, particiones, métricas y versiones quedan
  guardadas junto a los resultados.

La justificación de cada decisión y la forma en que se cumplen los principios de diseño están en
[docs/architecture.md](docs/architecture.md).
