# Arquitectura

Este documento describe las decisiones de diseño del repositorio y su justificación. Las decisiones se
agrupan según el criterio que las motiva: comparación justa entre grupos, trazabilidad,
reproducibilidad, análisis de sensibilidad y simplicidad. Al final se detalla cómo se cumple cada
principio de diseño.

## Flujo de un experimento

```
configs/<experimento>.yaml
        │
        ▼
InvoiceNetworkGenerator ──► InvoiceNetwork (empresas + facturas)
        │
        ├─► constructores de variables (transactional, structural) ──► vista tabular
        └─► InvoiceGraphBuilder.to_pyg ───────────────────────────────► vista de grafo
                                    │
                                    ▼
                             ExperimentData
                                    │
                    stratified_folds(y, n_splits, val_size, seed)
                                    │
        para cada pliegue, grupo y modelo:
            detector.fit(data, train_idx, val_idx)
            umbral = política.select(y[val_idx], detector.predict_proba(data, val_idx))
            compute_metrics(y[test_idx], detector.predict_proba(data, test_idx), umbral)
                                    │
                                    ▼
            results/runs/<AAAAMMDD-HHMMSS>_<experimento>/
```

## Comparación justa

La pregunta de investigación compara grupos que consumen la información de formas distintas (tablas
frente a grafos). Las diferencias en los resultados solo son atribuibles al enfoque si todo lo demás es
idéntico.

### Contenedor único `ExperimentData`

`src/efd/data.py` reúne la vista tabular (`features`, `labels`) y la vista de grafo (`graph`, objeto
`Data` de PyTorch Geometric). La posición `i` identifica a la misma empresa en ambas vistas: fila `i` de
la tabla y nodo `i` del grafo. Al construirse, el contenedor verifica que los identificadores sean
únicos, que las etiquetas estén alineadas con las variables, que el grafo tenga tantos nodos como filas
la tabla y que, si el grafo trae etiquetas, coincidan con las tabulares.

**Justificación:** si cada grupo cargara sus datos por separado, un reordenamiento o filtrado
accidental haría que G1, G2 y G3 se evaluaran sobre empresas distintas sin que ninguna prueba lo
detecte. Con un único contenedor la alineación se comprueba una sola vez y por construcción.

`select_features` deriva la vista de G1 a partir de la de G2 eliminando columnas, sin tocar etiquetas,
grafo ni orden de nodos.

### Particiones compartidas sobre nodos

`src/efd/evaluation/splits.py` genera un k-fold estratificado: cada pliegue actúa una vez como prueba y
el resto se divide, también de forma estratificada, en entrenamiento y validación. Las particiones son
posiciones de nodo y dependen únicamente de las etiquetas, de `n_splits`, de `val_size` y de la semilla.

**Justificación:**
- Al no depender de las variables ni del modelo, los mismos índices sirven a los tres grupos.
- La estratificación conserva la proporción de empresas fantasmas en cada partición; con clases
  desbalanceadas, una partición sin suficientes positivos haría inestable la AUC-PR.
- El k-fold permite reportar media y dispersión de cada métrica, en lugar de un único valor que podría
  depender de una partición favorable.
- La validación interna separa la selección del umbral y la parada temprana de la evaluación final.

Las GNN trabajan en modo transductivo: el grafo completo se propaga en el entrenamiento y solo las
etiquetas se restringen a los nodos de entrenamiento. Para mantener la igualdad de condiciones, las
métricas estructurales de G2 se calculan también sobre el grafo completo, que no contiene etiquetas.

### Interfaz común de detectores

`BaseDetector` (`src/efd/models/base.py`) define `fit(data, train_idx, val_idx)` y
`predict_proba(data, idx)`. Los métodos públicos validan que entrenamiento y validación no estén vacíos
ni se solapen y que la salida sea un arreglo 1-D en [0, 1] con un valor por nodo pedido; luego delegan
en `_fit` y `_predict_proba`, que implementa cada modelo.

**Justificación:** el orquestador trata igual a un Random Forest y a una GAT. Las validaciones en la
clase base impiden que un modelo concreto se salte las reglas comunes (por ejemplo, entrenar con nodos
de validación).

### Umbral elegido en validación

`src/efd/evaluation/threshold.py` elige el umbral sobre la partición de validación y lo aplica sin
cambios en prueba. La política por defecto (`max_f1`) toma el umbral que maximiza F1; ante empates
elige el más alto, que produce menos falsos positivos con el mismo F1. La regla de decisión
(`probabilidad >= umbral`) está en una única función, `binarize`.

**Justificación:** elegir el umbral en prueba inflaría F1, precisión y recall. Usar 0,5 para todos los
modelos penalizaría a los que producen probabilidades poco calibradas, algo frecuente con clases
desbalanceadas, y la diferencia reflejaría la calibración y no la capacidad de detección.

### Métricas

`src/efd/evaluation/metrics.py` es la única fuente de métricas. La principal es AUC-PR, estimada como
average precision. Se complementa con F1, precisión, recall y FPR (dependientes del umbral), precision@k
y AUC-ROC (independientes del umbral).

**Justificación:** con una proporción baja de empresas fantasmas, AUC-ROC y la exactitud se ven
dominadas por los verdaderos negativos; AUC-PR se centra en la clase positiva. Precision@k representa
una capacidad de fiscalización limitada a `k` empresas. Las métricas que no tienen sentido con una sola
clase (AUC-PR, AUC-ROC) lanzan un error en lugar de devolver un valor engañoso.

## Trazabilidad

Cada ejecución crea `results/runs/<AAAAMMDD-HHMMSS>_<experimento>/` (`src/efd/experiments/run_record.py`)
con la configuración efectiva, la semilla, los índices de cada pliegue (`Fold.to_dict`), las métricas
por grupo, modelo y pliegue en JSON y las versiones de Python y de las librerías.

**Justificación:** cualquier cifra reportada en el documento de titulación debe poder vincularse con la
configuración y el código que la produjeron. Las carpetas no se sobrescriben, así que las ejecuciones
anteriores quedan disponibles para comparación.

## Reproducibilidad

- La semilla se declara una sola vez (`experiment.seed` en el YAML) y se propaga al generador, a las
  particiones, a los modelos, a los explicadores y, mediante `src/efd/reproducibility.py`, a `random`,
  NumPy y PyTorch.
- Las particiones dependen solo de etiquetas y semilla; misma semilla, mismos índices (lo verifican las
  pruebas).
- Las versiones de las librerías se guardan con cada ejecución, porque un mismo código puede producir
  resultados distintos con versiones distintas.
- `pyproject.toml` declara el stack y `pytest` verifica los contratos antes de aceptar un cambio.

## Análisis de sensibilidad

Todos los parámetros que pueden influir en los resultados viven en la configuración y no en el código:
parámetros del generador (por ejemplo, la proporción de empresas fantasmas), número de pliegues, tamaño
de validación, política de umbral, hiperparámetros y semilla.

**Justificación:** el análisis de sensibilidad consiste en repetir el experimento variando un
parámetro y manteniendo el resto. Si ese parámetro estuviera en el código, cada variante exigiría
modificarlo; con la configuración basta un archivo nuevo, y la trazabilidad de cada ejecución permite
comparar las variantes después. La política `fixed` permite además estudiar cómo cambian las métricas
dependientes del umbral sin reentrenar.

## Simplicidad

- Sin bases de datos ni servicios externos: los datos se generan y se guardan en `data/`; los resultados,
  en archivos JSON y YAML.
- Un único paquete de Python (`efd`) con una carpeta por responsabilidad.
- Solo dos políticas de umbral y las métricas que exige la pregunta de investigación; el registro por
  nombre permite agregar más si llegan a necesitarse.

**Justificación:** el objetivo es responder una pregunta de investigación, no construir un producto.
Cada componente adicional es código que hay que probar y explicar en el documento.

## Principios de diseño

### S — Responsabilidad única

Cada módulo hace una sola cosa: `generation` simula la red, `graph` construye grafos y calcula métricas
estructurales, `features` produce variables, `models` entrena y predice, `evaluation` particiona, mide
y elige umbrales, `explainability` explica y `experiments` coordina y registra. Dentro de `evaluation`,
particiones, métricas y umbral están en módulos separados.

### O — Abierto/cerrado

Modelos, métricas, políticas de umbral, constructores de variables y explicadores se registran por
nombre (`src/efd/registry.py`) y la configuración los referencia por ese nombre. Agregar un modelo es
crear una clase que extienda `BaseDetector` con el decorador `@register_detector`; agregar una métrica es
decorar una función con `@register_metric`. Ni el orquestador ni `compute_metrics` cambian.

### L — Sustitución de Liskov

Todos los detectores aceptan las mismas particiones y devuelven puntuaciones con la misma forma y rango,
y la clase base lo verifica en cada llamada. Lo mismo ocurre con los explicadores: `BaseExplainer.explain`
comprueba que las atribuciones correspondan, fila a fila, a las empresas pedidas. Las pruebas instancian
cada detector registrado por su nombre y lo usan a través de la interfaz común.

### I — Segregación de interfaces

Las interfaces son pequeñas y separadas: `BaseDetector` (entrenar y predecir), `BaseExplainer`
(explicar), `BaseFeatureBuilder` (construir variables) y `ThresholdPolicy` (elegir umbral). Un detector
no está obligado a explicarse ni un constructor de variables a conocer los modelos.

### D — Inversión de dependencias

El orquestador depende de las abstracciones y de los registros, no de clases concretas: obtiene cada
componente con `create_detector`, `create_explainer`, `create_feature_builder` o
`create_threshold_policy` a partir del nombre en la configuración. Los modelos concretos dependen de
`ExperimentData` y `BaseDetector`, nunca del orquestador.

### DRY — Una sola fuente de verdad

- Métricas: `src/efd/evaluation/metrics.py`.
- Particiones: `src/efd/evaluation/splits.py`.
- Regla de decisión con umbral: `binarize` en `src/efd/evaluation/threshold.py`.
- Semilla: `experiment.seed` en la configuración, aplicada por `src/efd/reproducibility.py`.
- Mecánica de registro: la clase `Registry` en `src/efd/registry.py`, compartida por todos los registros.

### KISS — Lo mínimo necesario

Configuración en YAML, resultados en archivos, una clase base por tipo de componente y ninguna
infraestructura que no sea necesaria para responder la pregunta de investigación.
