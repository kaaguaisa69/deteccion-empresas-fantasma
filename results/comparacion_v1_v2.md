# Prueba de dificultad: v1 frente a v2

- v1: `20261112-114831_rf_g1_prev1_medio` sobre `data/synthetic/prev1_medio_s42_v1`
- v2: `20261113-101209_rf_g1_prev1_medio` sobre `data/synthetic/prev1_medio_s42`

## 1. Métricas (RF, G1, 5 pliegues, semilla 42)

| Métrica | v1 | v2 |
|---|---|---|
| config_hash | `b465da7f` | `bed8c385` |
| facturas | 1 995 109 | 2 012 073 |
| auc_pr | 0.9983 ± 0.0016 | 0.9432 ± 0.0076 |
| auc_roc | 1.0000 ± 0.0000 | 0.9988 ± 0.0004 |
| f1 | 0.9719 ± 0.0107 | 0.8603 ± 0.0394 |
| precision_at_100 | 1.0000 | 1.0000 |
| precision_at_200 | 1.0000 | 1.0000 |
| precision_at_400 | 0.9750 | 0.8750 |
| recall P1 (isolated_emitter) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| recall P2 (ring) | 0.9857 ± 0.0319 | 0.7360 ± 0.1526 |
| recall P3 (chain) | 0.9088 ± 0.0807 | 0.5363 ± 0.1937 |
| recall P4 (shared_representative) | 0.8640 ± 0.0397 | 0.7894 ± 0.1259 |

## 2. Veredicto (regla del 2026-10-05, lectura literal)

Demasiado fácil: AUC-PR ≥ 0,95, o recall ≥ 0,90 en P2, P3 y P4 a la vez. Demasiado difícil: AUC-PR ≤ 0,30. Adecuado: en otro caso.

- v1: **demasiado fácil** (AUC-PR 0.9983)
- v2: **adecuado** (AUC-PR 0.9432)

Hallazgos por patrón (recall medio ≥ 0,90):

- P2 (ring): v1 0.986 (sí); v2 0.736 (no)
- P3 (chain): v1 0.909 (sí); v2 0.536 (no)
- P4 (shared_representative): v1 0.864 (no); v2 0.789 (no)

## 3. Rasgos que delataban a las fantasmas en la v1 (medias)

| Rasgo | Grupo | v1 | v2 |
|---|---|---|---|
| age_days | legítimas | 3020.02 | 3020.02 |
| age_days | P2 (ring) | 592.54 | 2163.19 |
| age_days | P3 (chain) | 520.27 | 1993.21 |
| age_days | P4 (shared_representative) | 605.60 | 1717.19 |
| sales_mean_relative | legítimas | 1.18 | 1.25 |
| sales_mean_relative | P2 (ring) | 9.32 | 2.06 |
| sales_mean_relative | P3 (chain) | 9.56 | 1.92 |
| sales_mean_relative | P4 (shared_representative) | 2.63 | 3.13 |
| sales_counterparties | legítimas | 3.26 | 3.55 |
| sales_counterparties | P2 (ring) | 7.11 | 33.31 |
| sales_counterparties | P3 (chain) | 5.21 | 10.20 |
| sales_counterparties | P4 (shared_representative) | 6.06 | 5.95 |
| purchases_to_sales | legítimas | 3.60 | 3.37 |
| purchases_to_sales | P2 (ring) | -0.38 | -0.55 |
| purchases_to_sales | P3 (chain) | -2.03 | -0.82 |
| purchases_to_sales | P4 (shared_representative) | -1.62 | -1.61 |

## 4. Volumen de la v2 por patrón

| Patrón | Fantasmas | Facturas en que participan |
|---|---|---|
| P1 (isolated_emitter) | 160 | 11 064 |
| P2 (ring) | 80 | 15 942 |
| P3 (chain) | 80 | 3 831 |
| P4 (shared_representative) | 80 | 1 236 |
