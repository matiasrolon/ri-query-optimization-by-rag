# ri-query-optimization-by-rag

Comparison between baseline query optimizations and those implemented using RAG.

## Prerequisitos

### 1. Java JDK 11+

El proyecto utiliza [PyTerrier](https://github.com/terrier-org/pyterrier), que internamente ejecuta el motor de recuperación [Terrier IR](http://terrier.org/) (escrito en Java) a través de `pyjnius`. Por lo tanto, es **obligatorio** tener instalado un **Java Development Kit (JDK)** versión 11 o superior.

**Instalación en Ubuntu/Debian:**

```bash
sudo apt update && sudo apt install -y openjdk-11-jdk
```

**Verificar la instalación:**

```bash
java -version
javac -version
```

> [!NOTE]
> Si la variable de entorno `JAVA_HOME` no es detectada automáticamente, configurarla manualmente:
> ```bash
> export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
> ```
> Para hacerla persistente, agregar la línea anterior al archivo `~/.bashrc` o `~/.profile`.

### 2. Python 3.10+

El proyecto requiere **Python 3.10** o superior.

```bash
python3 --version
```

### 3. Colección MS MARCO Passage Ranking

El proyecto trabaja con la colección **MS MARCO Passage Ranking** de Microsoft. Los archivos del dataset deben descargarse manualmente y colocarse en el directorio `./data/`.

🔗 **Descargar desde:** [https://microsoft.github.io/msmarco/Datasets.html](https://microsoft.github.io/msmarco/Datasets.html)

#### Archivos necesarios

| Archivo | Descripción | Ubicación esperada |
|---|---|---|
| `collection.tsv` | Colección de ~8.8M passages (formato: `pid\tpassage`) | `./data/collection.tsv` |
| `queries.dev.small.tsv` | Queries de evaluación — dev/small split (6,980 queries) | `./data/queries.dev.small.tsv` |
| `qrels.dev.small.tsv` | Juicios de relevancia — dev/small split (7,437 juicios) | `./data/qrels.dev.small.tsv` |

> [!NOTE]
> Los archivos de queries y qrels son **opcionales**. Si no están presentes en `DATA_DIR`, se descargan automáticamente vía `ir_datasets` (`msmarco-passage/dev/small`) la primera vez que se ejecuta el benchmark.

> [!IMPORTANT]
> El archivo `collection.tsv` es indispensable para la indexación. Sin él, el programa no puede construir el índice.

#### Estructura esperada de directorios

```
data/                ← datos de entrada (dataset)
├── collection.tsv
├── queries.dev.small.tsv   ← opcional (se descarga automáticamente)
└── qrels.dev.small.tsv     ← opcional (se descarga automáticamente)

output/              ← generado automáticamente por el programa
└── index/
    └── ...
```

## Instalación

1. **Clonar el repositorio:**

```bash
git clone https://github.com/<usuario>/ri-query-optimization-by-rag.git
cd ri-query-optimization-by-rag
```

2. **Crear y activar un entorno virtual:**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**

```bash
cp .env.example .env
```

Editar `.env` según sea necesario. Las variables disponibles son:

**Indexación y rutas:**

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `INDEXING_METHOD` | Método de indexación (`pyterrier` o `pisa`) | `pyterrier` |
| `DATA_DIR` | Directorio de datos de entrada (dataset) | `./data` |
| `OUTPUT_DIR` | Directorio de salida para archivos generados | `./output` |
| `INDEX_DIR` | Directorio de índices generados | `./output/index` |
| `SAMPLE_FRACTION` | Fracción de la colección a indexar (0.0 - 1.0) | `1.0` |
| `FORCE_REINDEX` | Forzar re-indexación aunque ya exista un índice (`true`/`false`) | `false` |

**Parámetros de feedback (query expansion PRF y RAG):**

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `FEEDBACK_DOCS` | Cantidad de documentos usados como feedback para la expansión | `10` |
| `FEEDBACK_TERMS` | Cantidad de términos adicionales a agregar a la query (aplica a PRF y RAG) | `10` |
| `FEEDBACK_LAMBDA` | Peso de interpolación entre la query original y los términos de expansión (0.0 - 1.0) | `0.6` |
| `EXPANSION_MODEL` | Modelo DFR para PRF: `bo1` (Bose-Einstein 1) o `kl` (Kullback-Leibler divergence) | `bo1` |

**Configuración del LLM (expansión RAG):**

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `LLM_MODEL` | Modelo de lenguaje a utilizar (compatible con API OpenAI) | `qwen2.5:7b` |
| `LLM_BASE_URL` | URL base del servidor LLM (Ollama local u otro proveedor) | `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | API key para el proveedor LLM (dejar vacío para Ollama local) | *(vacío)* |

## Uso

```bash
python main.py
```

El programa detecta automáticamente si ya existe un índice construido en disco. Si existe, se reutiliza (a menos que `FORCE_REINDEX=true`).

### Parámetros opcionales

El script acepta los siguientes argumentos de línea de comandos para controlar la ejecución del benchmark:

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `--max-queries N` | Limitar la evaluación a las primeras N queries (útil para testing o ejecución por pasadas) | *(todas)* |
| `--offset N` | Saltar las primeras N queries antes de comenzar la evaluación | `0` |

**Ejemplos:**

```bash
# Ejecutar el benchmark completo
python main.py

# Evaluar solo las primeras 50 queries
python main.py --max-queries 50

# Evaluar queries 100 a 149 (útil para ejecución incremental)
python main.py --offset 100 --max-queries 50

# Procesar 4000 queries desde el inicio
python main.py --max-queries 4000 --offset 0
```

> [!TIP]
> Para datasets muy grandes (100k+ queries), se recomienda ejecutar el benchmark en pasadas incrementales usando `--offset` y `--max-queries`. Los resultados se exportan con timestamp en el nombre del archivo para evitar sobreescrituras.

## Módulos de Análisis de Resultados

El paquete `analysis` cuenta con dos modos de análisis:
1. **Análisis de una corrida particular** (`analyzer_single_run` / `run_single_analysis.py`).
2. **Análisis general multi-corrida** (`analyzer_summary` / `analyzer_summary.py`).

---

### 1. Análisis de una corrida en particular (`analyzer_single_run`)

Procesa el archivo CSV generado por una corrida individual del benchmark (`benchmark_results_<timestamp>.csv`) y genera visualizaciones (`.pdf` vectoriales y `.png` en alta resolución) junto con el reporte de métricas:

```bash
# Usando el script de corrida particular
python run_single_analysis.py --file "/ruta/a/benchmark_results_<timestamp>.csv"

# (Alias compatibles)
python run_analysis.py --file "/ruta/a/benchmark_results_<timestamp>.csv"
python -m analysis --file "/ruta/a/benchmark_results_<timestamp>.csv"
```

#### Gráficos y reportes generados (`output/figures/<nombre_del_archivo>/`)

1. **`01_gain_loss_per_query.pdf` (`.png`)**: Perfil continuo (waterfall) ordenado por $\Delta(q) = RR_{RAG}(q) - RR_{PRF}(q)$ con el conteo de consultas mejoradas, neutras y degradadas.
2. **`01b_gain_loss_donut.pdf` (`.png`)**: Proporción porcentual de consultas mejoradas, neutras y degradadas y balance neto.
3. **`02_time_breakdown_stacked.pdf` (`.png`)**: Desglose porcentual y absoluto de tiempos de ejecución (1ª pasada BM25, Text Fetch, Inferencia LLM y 2ª pasada).
4. **`03_lexicon_filtering_donut.pdf` (`.png`)**: Proporción de términos propuestos por el LLM aceptados vs. rechazados (alucinación fuera del lexicón, duplicados, stopwords, truncados).
5. **`04_terms_distribution_per_query.pdf` (`.png`)**: Distribución de cantidad de términos propuestos y aceptados por consulta.
6. **`05_llm_attempts_bar.pdf` (`.png`)**: Distribución de intentos de expansión del LLM (1, 2 o 3 intentos) y timeouts.
7. **`summary_metrics.json`** y **`summary_report.txt`**: Reporte detallado con MRR@10 (tanto sobre **todas las consultas** como **solo sobre consultas expandidas** para PRF y RAG), tiempos totales y tiempos promedio por consulta.

---

### 2. Análisis general multi-corrida (`analyzer_summary`)

Procesa una tabla CSV resumen con múltiples configuraciones de hiperparámetros (`FEEDBACK_DOCS` $\times$ `FEEDBACK_TERMS`, por defecto en `./data/benchmark_summary.csv`) y permite elegir si graficar con las columnas de **"Todas las queries"** (valor por defecto) o **"Solo expandidas"**.

```bash
# Ejecutar análisis general con "Todas las queries" (por defecto)
python analyzer_summary.py --file data/benchmark_summary.csv

# Ejecutar explícitamente con "Todas las queries"
python analyzer_summary.py --file data/benchmark_summary.csv --queries all

# Ejecutar con las columnas de "Solo expandidas"
python analyzer_summary.py --file data/benchmark_summary.csv --queries expanded

# También disponible como módulo de Python:
python -m analysis.analyzer_summary --file data/benchmark_summary.csv --queries all
```

#### Parámetros de `analyzer_summary`

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `--file`, `-f` | Ruta al archivo CSV de resumen general multi-corrida | `data/benchmark_summary.csv` |
| `--queries`, `-q` | Conjunto de consultas para calcular MRR@10: `all` / `todas` (**Todas las queries**) o `expanded` / `expandidas` (**Solo expandidas**) | `all` |
| `--output-dir`, `-o` | Directorio de salida para los gráficos y reportes | `output/figures/summary_todas/` (o `summary_solo_expandidas/`) |

#### Gráficos y reportes generados (`output/figures/summary_<modo>/`)

1. **`01_mrr_vs_fb_terms.pdf` (`.png`)**: **MRR@10 vs. `fb_terms`** — curvas de efectividad para RAG (LLM) y PRF según la cantidad de términos añadidos ($1, 2, 3, 5$) para cada valor de `fb_docs` ($3, 5, 10$), junto con la línea base de BM25.
2. **`02_mrr_vs_fb_docs.pdf` (`.png`)**: **MRR@10 vs. `fb_docs`** — curvas de efectividad en función de la cantidad de documentos de contexto ($3, 5, 10$) para cada valor de `fb_terms` ($1, 2, 3, 5$), junto con la línea base de BM25.
3. **`03_effectiveness_vs_cost.pdf` (`.png`)**: **Efectividad vs. costo** — gráfico de dispersión con el tiempo promedio por consulta en el eje X (**escala logarítmica**) y MRR@10 en el eje Y, mostrando un punto por cada configuración $(fb\_docs, fb\_terms)$ de RAG, PRF y la línea base BM25.
4. **`04_inference_time_vs_fb_docs.pdf` (`.png`)**: **Tiempo de inferencia vs. `fb_docs`** — muestra el crecimiento casi lineal del tiempo por consulta de RAG respecto a `fb_docs`, incluyendo las series por `fb_terms`, los promedios por punto y la recta de ajuste lineal ($R^2$).
5. **`summary_general_metrics.json`** y **`summary_general_report.txt`**: Resumen numérico y tabla comparativa de todas las configuraciones evaluadas.
