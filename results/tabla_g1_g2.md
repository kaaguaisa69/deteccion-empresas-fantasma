# G1 y G2 sobre la v2 (3 semillas × 5 pliegues = 15 mediciones)

| Modelo | AUC-PR global | AUC-PR sin P1 | F1 sin P1 | Recall P1 | Recall P2 | Recall P3 | Recall P4 | n |
|---|---|---|---|---|---|---|---|---|
| RF · G1 | 0.940 ± 0.017 | 0.864 ± 0.028 | 0.775 ± 0.055 | 1.000 ± 0.000 | 0.770 ± 0.141 | 0.575 ± 0.145 | 0.799 ± 0.114 | 15 |
| XGB · G1 | 0.962 ± 0.010 | 0.910 ± 0.019 | 0.821 ± 0.034 | 1.000 ± 0.000 | 0.852 ± 0.115 | 0.616 ± 0.136 | 0.895 ± 0.059 | 15 |
| RF · G2 | 0.976 ± 0.012 | 0.945 ± 0.025 | 0.866 ± 0.037 | 1.000 ± 0.000 | 0.911 ± 0.079 | 0.724 ± 0.132 | 0.895 ± 0.081 | 15 |
| XGB · G2 | 0.980 ± 0.008 | 0.955 ± 0.018 | 0.881 ± 0.037 | 0.994 ± 0.016 | 0.924 ± 0.107 | 0.752 ± 0.125 | 0.949 ± 0.067 | 15 |

Recall P1 es global; P2–P4 y F1 son sobre la población sin P1. Media ± desviación muestral.

## Corridas y tiempos (segundos)

| Modelo | Corrida | Duración total | Variables G1 por semilla | Métricas de red (G2) por semilla |
|---|---|---|---|---|
| RF · G1 | `20261113-152010_rf_g1_v2_3s` | 45 | 3.8 | — |
| XGB · G1 | `20261113-152053_xgb_g1_v2_3s` | 39 | 3.8 | — |
| RF · G2 | `20261113-152215_rf_g2_v2_3s` | 80 | 3.8 | 11.9 ± 0.3 |
| XGB · G2 | `20261113-152331_xgb_g2_v2_3s` | 72 | 3.8 | 11.8 ± 0.4 |

## Comprobación

RF · G1 con semilla 42 reproduce los pliegues de `20261113-101209_rf_g1_prev1_medio`: sí.
