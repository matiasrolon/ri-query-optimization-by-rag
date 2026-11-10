# Optimización de Consultas en Recuperación de Información mediante LLM

Comparación entre métodos clásicos de expansión de consultas (BM25 base y Pseudo-Relevance Feedback) y expansión asistida por Modelos Grandes de Lenguaje (LLM).

## Prerrequisitos

### 1. Java JDK 11+

El proyecto utiliza [PyTerrier](https://github.com/terrier-org/pyterrier), que internamente ejecuta el motor de recuperación [Terrier IR](http://terrier.org/) (escrito en Java) a través de `pyjnius`. Por lo tanto, es **obligatorio** tener instalado un **Kit de Desarrollo de Java (JDK)** versión 11 o superior.

**Instalación en Ubuntu/Debian:**

```bash
sudo apt update && sudo apt install -y openjdk-11-jdk
```

**Verificar la instalación:**

```bash
java -version
javac -version
```

> **Nota:**
> Si la variable de entorno `JAVA_HOME` no es detectada automáticamente, configurarla manualmente:
>
> ```bash
> export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
> ```
>
> Para hacerla persistente, agregar la línea anterior al archivo `~/.bashrc` o `~/.profile`.

### 2. Python 3.10+

El proyecto requiere **Python 3.10** o superior.

```bash
python3 --version
```

### 3. Colección MS MARCO Passage Ranking

El proyecto trabaja con la colección **MS MARCO Passage Ranking** de Microsoft. Los archivos del conjunto de datos deben descargarse manualmente y colocarse en el directorio `./data/`.

🔗 **Descargar desde:** [https://microsoft.github.io/msmarco/Datasets.html](https://microsoft.github.io/msmarco/Datasets.html)

#### Archivos necesarios


| Archivo                 | Descripción                                                        | Ubicación esperada            |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------ |
| `collection.tsv`        | Colección de ~8.8M pasajes (formato:`pid\tpassage`)                | `./data/collection.tsv`        |
| `queries.dev.small.tsv` | Consultas de evaluación — partición`dev/small` (6,980 consultas) | `./data/queries.dev.small.tsv` |
| `qrels.dev.small.tsv`   | Juicios de relevancia — partición`dev/small` (7,437 juicios)      | `./data/qrels.dev.small.tsv`   |
| `benchmark_summary.csv` | Tabla resumen de corridas para el análisis general multi-corrida   | `./data/benchmark_summary.csv` |

> **Nota:**
> Los archivos de consultas (`queries`) y juicios de relevancia (`qrels`) son **opcionales**. Si no están presentes en `DATA_DIR`, se descargan automáticamente vía `ir_datasets` (`msmarco-passage/dev/small`) la primera vez que se ejecuta la evaluación.

> **Importante:**
> El archivo `collection.tsv` es indispensable para la indexación. Sin él, el programa no puede construir el índice.

#### Estructura de directorios

```text
data/                                     ← datos de entrada
├── collection.tsv                        ← colección de pasajes MS MARCO
├── queries.dev.small.tsv                 ← opcional (se descarga automáticamente)
├── qrels.dev.small.tsv                   ← opcional (se descarga automáticamente)
└── benchmark_summary.csv                 ← tabla resumen de múltiples corridas

output/                                   ← generado automáticamente por el programa
├── index/                                ← índice invertido construido (PyTerrier / PISA)
│   └── ...
├── benchmark.log                         ← registro detallado de la ejecución
├── benchmark_results_<timestamp>.csv     ← resultados por consulta de cada corrida
└── figures/                              ← gráficos (.png) y reportes de análisis
    ├── benchmark_results_<timestamp>/    ← análisis de una corrida particular
    │   ├── 01_gain_loss_per_query.png
    │   ├── 01b_gain_loss_donut.png
    │   ├── 02_time_breakdown_stacked.png
    │   ├── 03_lexicon_filtering_donut.png
    │   ├── 04_terms_distribution_per_query.png
    │   ├── 05_llm_attempts_bar.png
    │   ├── summary_metrics.json
    │   └── summary_report.txt
    ├── summary_todas/                    ← análisis general (todas las consultas)
    │   ├── 01_mrr_vs_fb_terms.png
    │   ├── 02_mrr_vs_fb_docs.png
    │   ├── 03_effectiveness_vs_cost.png
    │   ├── 04_inference_time_vs_fb_docs.png
    │   ├── summary_general_metrics.json
    │   └── summary_general_report.txt
    └── summary_solo_expandidas/          ← análisis general (solo consultas expandidas)
        ├── 01_mrr_vs_fb_terms.png
        ├── 02_mrr_vs_fb_docs.png
        ├── 03_effectiveness_vs_cost.png
        ├── 04_inference_time_vs_fb_docs.png
        ├── summary_general_metrics.json
        └── summary_general_report.txt
```

## Instalación

1. **Clonar el repositorio:**

```bash
git clone <url-del-repositorio>
cd <directorio-del-proyecto>
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


| Variable          | Descripción                                                       | Valor por defecto |
| ----------------- | ------------------------------------------------------------------ | ----------------- |
| `INDEXING_METHOD` | Método de indexación (`pyterrier` o `pisa`)                      | `pyterrier`       |
| `DATA_DIR`        | Directorio de datos de entrada                                     | `./data`          |
| `OUTPUT_DIR`      | Directorio de salida para archivos generados                       | `./output`        |
| `INDEX_DIR`       | Directorio de índices generados                                   | `./output/index`  |
| `SAMPLE_FRACTION` | Fracción de la colección a indexar (0.0 - 1.0)                   | `1.0`             |
| `FORCE_REINDEX`   | Forzar re-indexación aunque ya exista un índice (`true`/`false`) | `false`           |

**Parámetros de retroalimentación (expansión de consultas PRF y LLM):**


| Variable          | Descripción                                                                                                             | Valor por defecto |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------- |
| `FEEDBACK_DOCS`   | Cantidad de documentos recuperados en primera pasada usados como contexto para la expansión                             | `10`              |
| `FEEDBACK_TERMS`  | Cantidad de términos adicionales a agregar a la consulta original (aplica a PRF y LLM)                                  | `10`              |
| `EXPANSION_MODEL` | Modelo de divergencia de la aleatoriedad (DFR) para PRF:`bo1` (Bose-Einstein 1) o `kl` (divergencia de Kullback-Leibler) | `bo1`             |

**Configuración del LLM (expansión mediante modelo de lenguaje):**


| Variable         | Descripción                                                        | Valor por defecto           |
| ---------------- | ------------------------------------------------------------------- | --------------------------- |
| `LLM_MODEL`      | Modelo de lenguaje a utilizar (compatible con API OpenAI)           | `qwen2.5:7b`                |
| `LLM_BASE_URL`   | URL base del servidor LLM (Ollama local u otro proveedor)           | `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | Clave de API para el proveedor LLM (dejar vacío para Ollama local) | *(vacío)*                  |

## Uso

```bash
python main.py
```

El programa detecta automáticamente si ya existe un índice construido en disco. Si existe, se reutiliza (a menos que `FORCE_REINDEX=true`).

### Parámetros opcionales

El script acepta los siguientes argumentos de línea de comandos para controlar la ejecución de la evaluación:


| Parámetro        | Descripción                                                                                    | Valor por defecto |
| ----------------- | ----------------------------------------------------------------------------------------------- | ----------------- |
| `--max-queries N` | Limitar la evaluación a las primeras N consultas (útil para pruebas o ejecución por pasadas) | *(todas)*         |
| `--offset N`      | Saltar las primeras N consultas antes de comenzar la evaluación                                | `0`               |

**Ejemplos:**

```bash
# Ejecutar la evaluación completa
python main.py

# Evaluar solo las primeras 50 consultas
python main.py --max-queries 50

# Evaluar consultas 100 a 149 (útil para ejecución incremental)
python main.py --offset 100 --max-queries 50

# Procesar 4000 consultas desde el inicio
python main.py --max-queries 4000 --offset 0
```

> **Consejo:**
> Para conjuntos de datos muy grandes, se recomienda ejecutar la evaluación en pasadas incrementales usando `--offset` y `--max-queries`. Los resultados se guardan en bloques incrementales y se exportan con marca de tiempo en el nombre del archivo (`output/benchmark_results_<timestamp>.csv`) para evitar sobreescrituras.

## Módulos de Análisis de Resultados

El paquete `analysis` cuenta con dos modos de análisis:

1. **Análisis de una corrida en particular** (`analyzer_single_run` / `run_single_analysis.py`).
2. **Análisis general multi-corrida** (`analyzer_summary` / `analyzer_summary.py`).

---

### 1. Análisis de una corrida en particular (`analyzer_single_run`)

Procesa el archivo CSV generado por una corrida individual (`benchmark_results_<timestamp>.csv`) y genera visualizaciones en alta resolución (`.png`) junto con el reporte de métricas:

```bash
# Usando el script de corrida particular
python run_single_analysis.py --file "/ruta/a/benchmark_results_<timestamp>.csv"

# (Alias compatibles)
python run_analysis.py --file "/ruta/a/benchmark_results_<timestamp>.csv"
python -m analysis --file "/ruta/a/benchmark_results_<timestamp>.csv"
```

#### Gráficos y reportes generados (`output/figures/<nombre_del_archivo>/`)

1. **`01_gain_loss_per_query.png`**: Perfil continuo de ganancias y pérdidas por consulta ordenado por $\Delta(q) = RR_{LLM}(q) - RR_{PRF}(q)$, indicando entre paréntesis la cantidad concreta de consultas mejoradas, neutras y degradadas.
2. **`01b_gain_loss_donut.png`**: Gráfico de anillo con la proporción porcentual de consultas mejoradas, neutras y degradadas y el balance neto.
3. **`02_time_breakdown_stacked.png`**: Desglose porcentual y absoluto de tiempos de ejecución (1ª pasada BM25, obtención de texto, inferencia LLM y 2ª pasada).
4. **`03_lexicon_filtering_donut.png`**: Proporción de términos propuestos por el LLM aceptados vs. rechazados (alucinación fuera del lexicón, duplicados, palabras vacías y truncados).
5. **`04_terms_distribution_per_query.png`**: Distribución de cantidad de términos propuestos y aceptados por consulta.
6. **`05_llm_attempts_bar.png`**: Distribución de intentos de expansión del LLM (1, 2 o 3 intentos) y excesos de tiempo límite.
7. **`summary_metrics.json`** y **`summary_report.txt`**: Reporte detallado con MRR@10 (calculado tanto sobre **todas las consultas** como **solo sobre consultas expandidas** para PRF y LLM), tiempos totales y tiempos promedio por consulta.

---

### 2. Análisis general multi-corrida (`analyzer_summary`)

Procesa una tabla CSV resumen con múltiples configuraciones de hiperparámetros (`FEEDBACK_DOCS` $\times$ `FEEDBACK_TERMS`, por defecto en `./data/benchmark_summary.csv`) y permite elegir si calcular los gráficos con las columnas de **"Todas las consultas"** (valor por defecto) o **"Solo expandidas"**.

```bash
# Ejecutar análisis general con todas las consultas (por defecto)
python analyzer_summary.py --file data/benchmark_summary.csv

# Ejecutar explícitamente con todas las consultas
python analyzer_summary.py --file data/benchmark_summary.csv --queries all

# Ejecutar solo con las consultas expandidas
python analyzer_summary.py --file data/benchmark_summary.csv --queries expanded

# También disponible como módulo de Python:
python -m analysis.analyzer_summary --file data/benchmark_summary.csv --queries all
```

#### Parámetros de `analyzer_summary`


| Parámetro           | Descripción                                                                                                    | Valor por defecto                                              |
| -------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `--file`, `-f`       | Ruta al archivo CSV de resumen general multi-corrida                                                            | `data/benchmark_summary.csv`                                   |
| `--queries`, `-q`    | Conjunto de consultas para calcular MRR@10:`all` (todas las consultas) o `expanded` (solo consultas expandidas) | `all`                                                          |
| `--output-dir`, `-o` | Directorio de salida para los gráficos y reportes                                                              | `output/figures/summary_todas/` (o `summary_solo_expandidas/`) |

#### Gráficos y reportes generados (`output/figures/summary_<modo>/`)

* **`01_mrr_vs_fb_terms.png`**: **MRR@10 vs. `fb_terms`** — curvas de efectividad para LLM y PRF según la cantidad de términos añadidos ($1, 2, 3, 5$) para cada valor de `fb_docs` ($3, 5, 10$), junto con la línea base de BM25.
* **`02_mrr_vs_fb_docs.png`**: **MRR@10 vs. `fb_docs`** — curvas de efectividad en función de la cantidad de documentos de contexto ($3, 5, 10$) para cada valor de `fb_terms` ($1, 2, 3, 5$), junto con la línea base de BM25.
* **`03_effectiveness_vs_cost.png`**: **Efectividad vs. costo** — curvas de MRR@10 en el eje Y en función del tiempo promedio por consulta en el eje X (**escala logarítmica**) para cada configuración $(fb\_docs, fb\_terms)$ de LLM, PRF y la línea base BM25.
* **`04_inference_time_vs_fb_docs.png`**: **Tiempo de inferencia vs. `fb_docs`** — muestra el crecimiento casi lineal del tiempo por consulta de LLM respecto a `fb_docs`, incluyendo las series por `fb_terms` y la recta de ajuste lineal ($R^2$).
* **`summary_general_metrics.json`** y **`summary_general_report.txt`**: Resumen numérico y tabla comparativa de todas las configuraciones evaluadas.
