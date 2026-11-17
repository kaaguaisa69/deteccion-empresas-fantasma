# G1 y G2 con y sin ajuste de hiperparámetros (v2, 3 semillas × 5 pliegues = 15)

## Con ajuste (≤ 12 combinaciones, elección por AUC-PR en validación)

| Modelo | AUC-PR global | AUC-PR sin P1 | F1 sin P1 | Recall P1 | Recall P2 | Recall P3 | Recall P4 | Combinación más elegida |
|---|---|---|---|---|---|---|---|---|
| RF · G1 | 0.940 ± 0.018 | 0.862 ± 0.031 | 0.772 ± 0.050 | 1.000 ± 0.000 | 0.737 ± 0.154 | 0.585 ± 0.143 | 0.808 ± 0.100 | class_weight=balanced_subsample, max_features=sqrt, min_samples_leaf=5 (6/15) |
| XGB · G1 | 0.971 ± 0.007 | 0.933 ± 0.016 | 0.842 ± 0.047 | 1.000 ± 0.000 | 0.899 ± 0.121 | 0.664 ± 0.124 | 0.886 ± 0.083 | colsample_bytree=0.8, max_depth=3, min_child_weight=1, subsample=0.8 (5/15) |
| RF · G2 | 0.975 ± 0.012 | 0.942 ± 0.026 | 0.865 ± 0.043 | 0.998 ± 0.007 | 0.900 ± 0.102 | 0.711 ± 0.114 | 0.899 ± 0.081 | class_weight=balanced_subsample, max_features=sqrt, min_samples_leaf=5 (4/15) |
| XGB · G2 | 0.986 ± 0.008 | 0.967 ± 0.016 | 0.901 ± 0.037 | 0.993 ± 0.017 | 0.935 ± 0.103 | 0.765 ± 0.132 | 0.958 ± 0.045 | colsample_bytree=1.0, max_depth=3, min_child_weight=1, subsample=1.0 (5/15) |

## Sin ajuste (13/11)

| Modelo | AUC-PR global | AUC-PR sin P1 | F1 sin P1 | Recall P1 | Recall P2 | Recall P3 | Recall P4 | Combinación más elegida |
|---|---|---|---|---|---|---|---|---|
| RF · G1 | 0.940 ± 0.017 | 0.864 ± 0.028 | 0.775 ± 0.055 | 1.000 ± 0.000 | 0.770 ± 0.141 | 0.575 ± 0.145 | 0.799 ± 0.114 | fija |
| XGB · G1 | 0.962 ± 0.010 | 0.910 ± 0.019 | 0.821 ± 0.034 | 1.000 ± 0.000 | 0.852 ± 0.115 | 0.616 ± 0.136 | 0.895 ± 0.059 | fija |
| RF · G2 | 0.976 ± 0.012 | 0.945 ± 0.025 | 0.866 ± 0.037 | 1.000 ± 0.000 | 0.911 ± 0.079 | 0.724 ± 0.132 | 0.895 ± 0.081 | fija |
| XGB · G2 | 0.980 ± 0.008 | 0.955 ± 0.018 | 0.881 ± 0.037 | 0.994 ± 0.016 | 0.924 ± 0.107 | 0.752 ± 0.125 | 0.949 ± 0.067 | fija |

Recall P1 es global; F1 y recall P2–P4 son sin P1. Media ± desviación muestral.

## Distribución de la combinación elegida (índice en la rejilla: pliegues)

- RF · G1: 0: 5, 1: 1, 3: 6, 6: 1, 7: 2
- XGB · G1: 0: 5, 1: 3, 2: 4, 3: 2, 4: 1
- RF · G2: 0: 3, 1: 2, 2: 3, 3: 4, 5: 3
- XGB · G2: 0: 3, 1: 5, 2: 2, 3: 3, 6: 1, 11: 1

## Línea base definitiva de H1

- Mejor G1 (AUC-PR sin P1 medio): XGB · G1, 0.9329; error = 0.0671.
- G3 debe alcanzar un AUC-PR sin P1 ≥ 0.9429 (error ≤ 0.0571) y Wilcoxon significativo.
- Mejor G2: XGB · G2, 0.9666; reducción descriptiva del error frente al mejor G1: 50.3% (sin prueba estadística).

## Corridas

- RF · G1: `20261116-100528_rf_g1_v2_ajuste` (621 s)
- XGB · G1: `20261116-101129_xgb_g1_v2_ajuste` (355 s)
- RF · G2: `20261116-102127_rf_g2_v2_ajuste` (593 s)
- XGB · G2: `20261116-102615_xgb_g2_v2_ajuste` (282 s)
