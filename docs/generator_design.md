# Diseño del generador de redes sintéticas de facturación electrónica

## Propósito
Generar una red de facturación electrónica entre contribuyentes, inspirada en el esquema ecuatoriano de
comprobantes electrónicos, con empresas fantasmas inyectadas bajo patrones conocidos y niveles controlados
de desbalance y camuflaje. La red sustituye a los datos reales, que están protegidos por la reserva tributaria
(art. 99 del Código Tributario).

Todo parámetro que no proviene de una fuente se marca en la configuración como `supuesto de diseño`.

## Escala base
| Parámetro | Valor |
|---|---|
| Contribuyentes | 40 000 |
| Prevalencia de empresas fantasmas | 1 % (400) |
| Período simulado | 12 meses (2025-01-01 a 2025-12-31) |
| Facturas por contribuyente al año (media) | 50 (~2 000 000 comprobantes) |
| Tarifa de IVA | 15 % (con una fracción de bienes con tarifa 0 %) |

## Modelo de datos (red heterogénea)
- **Nodo contribuyente**: tipo (sociedad / persona natural), sector, provincia, fecha de inicio de actividades,
  tamaño (micro / pequeña / mediana / grande).
- **Nodo representante legal**: vínculo societario entre contribuyentes.
- **Arista factura** (emisor → receptor): comprobantes individuales, agregables por par.
- **Arista representación** (representante → contribuyente).

## Comportamiento legítimo
1. **Conexión preferencial**: pocos contribuyentes concentran muchos clientes y la mayoría tiene pocos.
2. **Matriz intersectorial**: probabilidad de compra entre sectores (p. ej., construcción compra a manufactura y comercio).
3. **Montos**: distribución lognormal dependiente del tamaño; estacionalidad mensual.
4. **IVA**: 15 % sobre el subtotal gravado; una fracción de bienes con tarifa 0 %.
5. **Ruido legítimo** (evita señales triviales):
   - grupos empresariales legítimos que comparten representante legal;
   - comercio recíproco entre empresas legítimas, que genera ciclos que no son fraude;
   - ciclos legítimos de comercio de 3 a 6 empresas, para que un ciclo de esa longitud no implique fraude;
   - una fracción de contribuyentes legítimos (supuesto de diseño) con `start_date` dentro del período
     simulado, para que la antigüedad no delate a las fantasmas;
   - una fracción de facturas legítimas (supuesto de diseño) con montos redondos, por la misma razón.

## Patrones de empresas fantasmas
Solo el emisor fantasma recibe la etiqueta `is_shell = 1`. Las empresas beneficiarias, que compran facturas
falsas, son contribuyentes legítimos y no se etiquetan.

| Patrón | Comportamiento | Proporción |
|---|---|---|
| P1 Emisor aislado | Vida corta (3–9 meses), vende sin compras proporcionales, montos redondos, muchos clientes beneficiarios | 40 % |
| P2 Anillo circular | Grupos de 3–6 fantasmas que se facturan en ciclo | 20 % |
| P3 Cadena de intermediarios | Fantasma → fantasma → beneficiario, de 2 a 4 saltos | 20 % |
| P4 Representante compartido | Grupos de 3–5 fantasmas con el mismo representante legal | 20 % |

P1 deja señal tabular fuerte; P2, P3 y P4 dejan señal principalmente relacional. Los resultados se reportan
por patrón para evidenciar qué enfoque detecta cada uno.

## Camuflaje
Proporción de las transacciones de cada fantasma que imitan comercio normal (contrapartes, montos y fechas
legítimas):

| Nivel | Proporción |
|---|---|
| Bajo | 10 % |
| Medio (base) | 30 % |
| Alto | 50 % |

## Escenarios
Prevalencia {0,5 %, 1 %, 2 %} × camuflaje {bajo, medio, alto} = 9 escenarios. Escenario base: 1 % y medio.

## Calibración con datos públicos
- Distribución provincial de fantasmas concentrada en Guayas, Pichincha y El Oro (reporte periodístico citado en la tesis).
- Distribución sectorial a partir del catastro público de empresas fantasmas del SRI, usado solo en forma agregada.

## Módulos (`src/efd/generation/`)
| Módulo | Responsabilidad |
|---|---|
| `__main__.py` | Comando `python -m efd.generation --config configs/generator.yaml` |
| `config.py` | `GeneratorConfig` (dataclass) cargada y validada desde YAML |
| `population.py` | Contribuyentes y representantes legales |
| `market.py` | Facturación legítima (conexión preferencial, matriz intersectorial, montos, IVA, estacionalidad) |
| `patterns/base.py` | `FraudPattern` (ABC): `name`, `inject(state, rng)`; `GenerationState`, el estado que recibe `inject` |
| `patterns/isolated_emitter.py` | P1 |
| `patterns/ring.py` | P2 |
| `patterns/chain.py` | P3 |
| `patterns/shared_representative.py` | P4 |
| `patterns/registry.py` | Registro de patrones por nombre (agregar uno nuevo no modifica el resto) |
| `camouflage.py` | Aplicación del nivel de camuflaje |
| `validation.py` | Comprobaciones de realismo e integridad |
| `writer.py` | Escritura de tablas y metadatos |
| `generator.py` | Orquestación: población → mercado → patrones → camuflaje → validación → escritura |

`InvoiceNetwork`, el contenedor de las tablas (esquema, nombres de archivo y lectura), está en
`src/efd/network.py`, fuera de `generation/`, porque es el contrato que comparten la generación, el grafo y
los constructores de variables.

Reglas:
- Un único `numpy.random.Generator` creado desde la semilla y pasado explícitamente; sin aleatoriedad global.
- Operaciones vectorizadas (numpy/pandas) para que el escenario base se genere en pocos minutos en un portátil.

## Salidas (`data/synthetic/<escenario>/`)
Las tablas se escriben en Parquet (requiere `pyarrow`). El nombre del escenario combina prevalencia, nivel
de camuflaje y semilla con el formato `prev<prevalencia>_<camuflaje>_s<semilla>`; por ejemplo, `prev1_medio_s42`.
La huella de la configuración (`config_hash`) excluye `output_root`, que no altera los datos y cambia de forma
según el sistema operativo.

| Archivo | Columnas |
|---|---|
| `taxpayers.parquet` | `taxpayer_id`, `taxpayer_type`, `sector`, `province`, `start_date`, `size`, `is_shell`, `pattern` |
| `invoices.parquet` | `invoice_id`, `issuer_id`, `receiver_id`, `issue_date`, `subtotal_0`, `subtotal_15`, `vat`, `total` |
| `representatives.parquet` | `representative_id`, `taxpayer_id` |
| `metadata.json` | Configuración usada, semilla, hash de la configuración, conteos, versiones de librerías |

`is_shell` y `pattern` son etiquetas: ningún módulo de variables puede usarlas como entrada.

## Validaciones (pruebas automáticas)
1. Misma semilla y configuración producen archivos idénticos.
2. El número de fantasmas es exactamente el configurado y cada patrón respeta su proporción.
3. Cada anillo P2 forma un ciclo; cada cadena P3 tiene entre 2 y 4 saltos; cada grupo P4 comparte representante.
4. No existen autofacturas; `vat = round(0,15 × subtotal_15, 2)`; `total = subtotal_0 + subtotal_15 + vat`.
5. Toda factura cae dentro del período y después de la fecha de inicio del emisor; un P1 no factura fuera de su vida.
   La fecha de fin de la vida de un P1 no está en el esquema: al generar se valida con la vida simulada, y sobre los
   archivos, con `start_date` más la vida máxima configurada.
6. La distribución de grado es de cola pesada (grado máximo mayor que 10 veces la mediana).
7. Existen grupos legítimos con representante compartido, comercio recíproco legítimo y ciclos legítimos de tres o
   más contribuyentes.
8. Existen contribuyentes legítimos con `start_date` dentro del período, al menos tantos como fantasmas con
   `start_date` dentro del período.
9. Existen facturas legítimas con montos redondos, al menos tantas como facturas emitidas por fantasmas con montos
   redondos.