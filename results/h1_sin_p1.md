# Línea base G1 sin P1 (corrida v2, sin reentrenar)

- Corrida: `20261113-101209_rf_g1_prev1_medio`; escenario `data/synthetic/prev1_medio_s42` (`config_hash bed8c385`).
- Mediciones: 5 (5 pliegues × 1 semilla). La regla de H1 usa 15 (5 × 3 semillas): valor preliminar.
- Población sin P1: legítimos + P2 (ring), P3 (chain) y P4 (shared_representative); umbral de cada pliegue.

## Comprobación: métricas globales recalculadas frente a metrics.json

| Métrica | recalculada | metrics.json |
|---|---|---|
| auc_pr | 0.9432 | 0.9432 |
| f1 | 0.8603 | 0.8603 |

## Resultado

| Métrica | Global (con P1) | Sin P1 |
|---|---|---|
| auc_pr | 0.9432 ± 0.0076 | 0.8679 ± 0.0260 |
| f1 | 0.8603 ± 0.0394 | 0.7564 ± 0.0586 |
| recall ring | — | 0.7360 ± 0.1526 |
| recall chain | — | 0.5363 ± 0.1937 |
| recall shared_representative | — | 0.7894 ± 0.1259 |

## Criterio de H1 que aplica

Línea base sin P1 = 0.8679 ≤ 0,87: el criterio es la mejora relativa ≥ 15 %, es decir, AUC-PR de G3 ≥ 0.9980 (y Wilcoxon significativo).
