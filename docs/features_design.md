# Diseño de variables, grafo, modelos tabulares y orquestador

## Regla de separación entre grupos
- **G1 (tabular):** solo información calculable con los registros propios del contribuyente: su ficha y
  sus facturas emitidas y recibidas. Equivale a una consulta SQL agrupada por contribuyente.
- **G2:** las variables de G1 más métricas que solo existen al analizar la red.
- **Prohibido en ambos:** cualquier variable derivada de las etiquetas (`is_shell`, `pattern`) o de la
  etiqueta de los vecinos. Una prueba automática permuta las etiquetas y exige que las variables de G1 y
  G2 no cambien.

Las métricas de red se calculan en `src/efd/graph/`; `src/efd/features/` solo las consume.

## Convenciones comunes
- **Montos:** base sin IVA, `subtotal_0 + subtotal_15`.
- **Monto redondo:** múltiplo de `round_multiple` (100 por defecto; configurable en el YAML del
  experimento).
- **Período:** se lee de `metadata.json` del escenario. Los constructores de variables lo reciben en un
  `FeatureContext` junto con la semilla de la corrida, de modo que el orquestador no conoce los
  parámetros particulares de cada constructor.
- **Valores vacíos:**
  - Sin ventas (o sin compras), las variables de ese bloque valen 0.
  - Con menos de dos facturas, los días medios entre facturas valen la duración del período en días.
  - Las relaciones entre dos montos se expresan como `log((1 + a) / (1 + b))`, finita aunque alguno
    sea 0.
- **Escenario transductivo:** las variables usan todas las facturas del período, incluidas las de los
  contribuyentes que caen en la partición de prueba, pero nunca sus etiquetas.

## Variables G1 por contribuyente
| Bloque | Variable | Definición |
|---|---|---|
| Ficha | `is_company` | 1 si `taxpayer_type` es `sociedad` |
| | `size_rank` | micro = 0, pequena = 1, mediana = 2, grande = 3 |
| | `sector_<s>`, `province_<p>` | codificación 0/1, una columna por categoría |
| | `age_days` | días desde `start_date` hasta el fin del período |
| Ventas (`sales_`) y compras (`purchases_`), por separado | `count` | número de facturas |
| | `total` | suma de montos |
| | `mean`, `max`, `std` | media, máximo y desviación (poblacional) de los montos |
| | `round_share` | proporción de facturas con monto redondo |
| | `zero_rated_share` | proporción de facturas con `subtotal_0 > 0` |
| | `counterparties` | contrapartes distintas |
| | `herfindahl` | Σ (participación del monto por contraparte)² |
| Relaciones | `purchases_to_sales` | `log((1 + compras) / (1 + ventas))` |
| | `vat_paid_to_collected` | `log((1 + IVA pagado) / (1 + IVA cobrado))` |
| Relativas | `sales_mean_relative`, `purchases_mean_relative` | monto medio ÷ mediana de los montos medios de los contribuyentes del mismo sector y tamaño que tienen facturas de ese tipo |
| Temporales | `active_months` | meses con alguna factura emitida o recibida |
| | `peak_month_sales_share` | ventas del mes con más ventas ÷ ventas totales |
| | `mean_days_between_invoices` | media de los días entre facturas consecutivas, emitidas y recibidas juntas |

## Variables G2 (además de G1)
| Variable | Grafo | Definición |
|---|---|---|
| `pagerank` | dirigido emisor → receptor | PageRank ponderado por el monto total del par |
| `clustering` | no dirigido | coeficiente de clustering local |
| `triangles` | no dirigido | número de triángulos que contienen al nodo |
| `core_number` | no dirigido | k-core máximo al que pertenece el nodo |
| `community_size`, `community_density` | no dirigido | tamaño y densidad de su comunidad Louvain, ponderada por número de facturas y calculada con la semilla de la corrida |
| `reciprocity` | dirigido | proporción de sus aristas que tienen la arista inversa |
| `scc_size` | dirigido | tamaño de su componente fuertemente conexo |
| `shared_representative_count` | representación | número de otras empresas con las que comparte representante legal |

## Grafo para G3 (`HeteroData`)
- **Nodos `taxpayer`:** `x` = variables de G1 sin escalar; `y` = etiqueta; mismo orden que la vista
  tabular.
- **Nodos `representative`:** `x` = constante 1.
- **Aristas de factura:** `(taxpayer, invoices, taxpayer)` y su inversa `(taxpayer, invoiced_by,
  taxpayer)`, agregadas por par. Sus atributos son el número de facturas, `log(1 + monto total)` y la
  primera y la última fecha normalizadas a [0, 1] dentro del período.
- **Aristas de representación:** `(representative, represents, taxpayer)` y su inversa `(taxpayer,
  represented_by, representative)`.
- **Escalado:** al entrenar las GNN, el escalado de las variables de los nodos `taxpayer` se ajusta solo
  con los nodos de entrenamiento de cada pliegue.

## Modelos tabulares
- **`random_forest`:** `RandomForestClassifier` con `class_weight` del YAML para el desbalance. La
  partición de validación no se usa para entrenar.
- **`xgboost`:** `XGBClassifier`. Si el YAML no fija `scale_pos_weight`, se calcula como negativos ÷
  positivos de entrenamiento. La partición de validación se usa para la parada temprana con `aucpr`.

## Modelos neuronales (G3 y control MLP)
- **D1 — Variables de nodo:** las variables de G1 (`feature_set: G1`) son los atributos de los nodos `taxpayer`
  en G3 y las entradas del MLP. Así, G3 frente a G1 aísla el aporte del grafo, y G3 frente a G2 compara el grafo
  aprendido con las métricas de red hechas a mano.
- **D2 — Preprocesamiento de nodos:** primero, una transformación fija sin ajuste, `signo(x)·log(1+|x|)`, porque
  hay montos de hasta 10⁸. Después, estandarización con la media y la desviación de los nodos de entrenamiento;
  una desviación de 0 se toma como 1. Supuesto de diseño.
- **D3 — Atributos de arista de GAT:** las cuatro columnas de `edge_attr` son el número de facturas (con
  `log(1+x)`), el monto ya guardado como `log(1+monto)`, y la primera y la última fecha en [0, 1]. Las cuatro se
  estandarizan con todas las aristas. Las aristas no llevan etiquetas, así que no hay fuga, y el grafo completo
  es observable en el modo transductivo. Supuesto de diseño. GraphSAGE no lee `edge_attr`.
- **D4 — Arquitectura:**
  - Una proyección lineal por tipo de nodo a la dimensión oculta (`taxpayer` 60 → h; `representative` 1 → h).
  - Después, `num_layers` capas `HeteroConv(aggr="sum")` sobre los 4 tipos de arista, cada una con activación y
    dropout: ReLU en GraphSAGE y ELU en GAT.
  - GraphSAGE usa `SAGEConv((-1, -1), h, aggr="mean")` con su raíz propia.
  - GAT usa `GATConv((-1, -1), hidden_channels, heads, concat=True, add_self_loops=False)`, con `edge_dim=4` en
    las aristas de factura y sin `edge_dim` en las de representación. Lleva además un término raíz lineal por
    tipo de nodo, porque sin lazos propios el nodo perdería su información. Su dimensión oculta es
    `heads × hidden_channels`.
  - Al final, una capa lineal h → 1 sobre `taxpayer` produce el logit.
  - El MLP tiene `num_layers` capas lineales, de 60 → h hasta h → 1, con ReLU y dropout entre ellas.
  - Inicialización perezosa: `SAGEConv` y `GATConv` crean sus pesos en la primera pasada. `HeteroGNN` hace esa
    pasada con `torch.no_grad()`, después de `torch.manual_seed(seed)` y antes de crear Adam.
  - Capacidad: con el mismo `num_layers`, el MLP tiene menos capas y no hace mensajes, así que su capacidad no
    es igual a la de las GNN.
- **D5 — Entrenamiento:**
  - Grafo completo en cada época, con un paso de Adam (`lr`, `weight_decay`).
  - Pérdida `BCEWithLogitsLoss` con `pos_weight` = legítimos ÷ fantasmas de entrenamiento, calculada solo sobre
    los nodos de entrenamiento.
  - Al final de cada época, AUC-PR de validación en modo `eval`. Si supera estrictamente al mejor, se guardan
    los pesos de esa época.
  - Parada temprana tras 20 épocas sin mejora, o al llegar a `epochs`. Al terminar se restauran los mejores pesos.
  - Las etiquetas se leen solo desde `data.y`.
- **D6 — Cómputo:** CPU, no GPU. Con `torch.use_deterministic_algorithms(True)`, `torch.manual_seed(seed)` al
  inicio de cada ajuste y `num_threads` fijo (`torch.set_num_threads`), la misma semilla da el mismo resultado en
  el mismo equipo y entorno. La clave `torch_num_threads` de `metrics.json` registra los hilos usados. Limitación:
  entre equipos o con otro número de hilos, los decimales pueden diferir.

## Evaluación
- **Por pliegue:** AUC-PR, AUC-ROC, F1, precisión, recall, FPR y recall por patrón, con el umbral elegido
  en validación. El resumen reporta media ± desviación estándar muestral entre pliegues.
- **Subpoblaciones (opcional):** la clave `evaluation.subpopulations` del YAML, `{nombre: [patrón excluido, ...]}`
  (por ejemplo `sin_p1: [isolated_emitter]`), agrega por pliegue las mismas métricas y el recall por patrón
  sin los fantasmas de esos patrones, con el mismo umbral del pliegue. Quedan en
  `folds[i].subpopulations.<nombre>` y en `summary.subpopulations.<nombre>`; sin la clave, no aparecen.
  No se calcula precision@k por subpoblación.
- **precision@k, sobre las predicciones de todos los pliegues juntas, por semilla:** cada contribuyente
  está en prueba en exactamente un pliegue, así que la unión de los pliegues de prueba cubre la población
  completa (40 000 contribuyentes y 400 fantasmas en el escenario base). Se ordena por puntuación y se
  mide la proporción de fantasmas entre las `k` primeras; no usa umbral.

  **Justificación:** por pliegue de prueba hay unas 80 fantasmas, así que precision@400 tendría un
  máximo de 0,20 y no mediría la capacidad del modelo.
- **Recall por patrón:** la columna `pattern` se entrega solo al evaluador y nunca entra en
  `ExperimentData`, de modo que ningún detector puede leerla.

## Orquestador
- **Comando:** `python -m efd.experiments --config configs/experiments/<experimento>.yaml`.
- **Contenido del YAML:** escenario, grupo de variables (`G1` o `G2`), modelo, número de particiones y
  lista de semillas. Cada semilla repite el k-fold completo con sus propias particiones, modelo y
  comunidades Louvain.
- **Ajuste de hiperparámetros (opcional):** la clave `model.grid` del YAML es una lista de 1 a 12
  combinaciones distintas y no vacías (`MAX_GRID_SIZE`), cada una un diccionario que se fusiona sobre
  `model.params`. En cada pliegue se entrena un detector por combinación (con la semilla de la corrida) y se
  elige el de mayor AUC-PR en validación (empate: índice menor); ese detector, sin reentrenar, fija el umbral
  en validación y predice la prueba, que no interviene en la elección. `folds[i].tuning` guarda
  `{selected, params, val_auc_pr}` y `summary.tuning.selected_counts` cuenta cuántos pliegues eligieron cada
  índice. Sin `grid`, la corrida es la de siempre y no aparece `tuning`.
- **Carpeta de cada corrida:** `results/runs/<AAAAMMDD-HHMMSS>_<experimento>/` con estos archivos:

| Archivo | Contenido |
|---|---|
| `config.yaml` | configuración usada |
| `scenario.json` | ruta del escenario y su `config_hash` |
| `folds.json` | índices de nodo de cada pliegue, por semilla |
| `metrics.json` | métricas por semilla y pliegue, subpoblaciones (si se piden), `tuning` por pliegue (si hay `model.grid`), precision@k por semilla, `feature_seconds` (segundos de cada bloque de variables, una entrada por semilla) y resumen (con `summary.subpopulations`, `summary.feature_seconds` y `summary.tuning`) |
| `predictions.parquet` | identificador, semilla, pliegue, puntuación, umbral y etiqueta de cada predicción de prueba |
| `environment.json` | versiones de Python y de las librerías |
| `run_info.json` | commit del código con que se ejecutó, si había cambios sin commit y hora de inicio |
| `progress/` | un archivo por pliegue terminado (se guarda al instante); si la corrida se interrumpe, al relanzarla con la misma configuración y el mismo commit continúa desde el siguiente pliegue pendiente |
