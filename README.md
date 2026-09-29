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

## Módulo de Análisis de Resultados

El paquete `analysis` permite procesar un archivo CSV generado por una ejecución del benchmark y generar automáticamente visualizaciones y reportes detallados en alta resolución:

```bash
# Ejecutar análisis sobre un archivo de resultados
python run_analysis.py --file "/ruta/a/benchmark_results_<timestamp>.csv"

# O como módulo Python:
python -m analysis --file "/ruta/a/benchmark_results_<timestamp>.csv"
```

### Gráficos generados

Las figuras se almacenan automáticamente en `./figures/<nombre_del_archivo>/`:

1. **`01_gain_loss_per_query.png`**: Análisis de ganancias y pérdidas por consulta.
   - Perfil continuo (waterfall) ordenado por $\Delta(q) = RR_{RAG}(q) - RR_{PRF}(q)$.
   - Histograma de distribución de diferencias y balance neto de consultas mejoradas, neutras y degradadas.
2. **`02_time_breakdown_stacked.png`**: Barras apiladas del desglose de tiempos de ejecución.
   - Comparación PRF vs RAG (1ª pasada BM25, Text Fetch, Inferencia LLM, 2ª pasada).
   - Vista absoluta (segundos) y porcentual (100% stacked).
   - Cálculo destacado del **tiempo promedio de inferencia LLM** del archivo.
3. **`03_lexicon_filtering_hallucination.png`**: Filtrado de lexicón y alucinación del LLM.
   - Ciclo de vida de términos (originales, propuestos por LLM, aceptados en el índice y expandidos).
   - Cuantificación de la tasa de alucinación (términos fuera del vocabulario de la colección) vs tasa de supervivencia.
4. **`summary_metrics.json`** y **`summary_report.txt`**: Resumen numérico y reporte textual con todas las métricas agregadas.
