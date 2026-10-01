# -*- coding: utf-8 -*-
"""
Command-line interface for the benchmark analysis module.

Usage:
    python -m analysis --file /path/to/benchmark_results_*.csv
    python run_analysis.py --file /path/to/benchmark_results_*.csv
"""

from __future__ import annotations

import argparse
import os
import sys

from analysis.analyzer import BenchmarkAnalyzer
from analysis.plots import generate_all_plots


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analizador de resultados de benchmarks de Recuperación de Información (RAG vs PRF)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Ruta al archivo CSV de resultados (ej. benchmark_results_20260920_230342.csv)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directorio base para guardar las figuras (por defecto: output/figures/<nombre_archivo>/)."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"❌ Error: El archivo especificado no existe:\n   {args.file}", file=sys.stderr)
        return 1

    file_stem = os.path.splitext(os.path.basename(args.file))[0]
    out_dir = args.output_dir or os.path.join("output", "figures", file_stem)

    print("\n" + "=" * 70)
    print(" 📊 MÓDULO DE ANÁLISIS: QUERY EXPANSION BENCHMARK")
    print("=" * 70)
    print(f"  • Archivo analizado : {args.file}")
    print(f"  • Directorio salida : {os.path.abspath(out_dir)}")
    print("=" * 70 + "\n")

    print("⏳ Procesando datos y calculando métricas...")
    try:
        analyzer = BenchmarkAnalyzer(args.file)
        metrics, fig_paths = generate_all_plots(analyzer, output_dir=out_dir)
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "─" * 70)
    print(" 📈 RESUMEN DE RESULTADOS")
    print("─" * 70)
    print(f"  • Consultas comunes evaluadas        : {metrics.common_query_count:,}")
    print(f"  • MRR@10 RAG (LLM)                  : {metrics.mrr_rag_mean:.4f}")
    print(f"  • MRR@10 PRF                        : {metrics.mrr_prf_mean:.4f}")
    print(f"  • Diferencia neta MRR (RAG − PRF)   : {metrics.mrr_diff:+.4f} ({metrics.mrr_relative_gain_pct:+.2f}%)")
    print()
    print("  Ganancias y Pérdidas por Consulta:")
    print(f"    - Mejoradas (RAG > PRF)           : {metrics.gain_loss.improved_count:,} ({metrics.gain_loss.improved_pct:.2f}%)")
    print(f"    - Neutras   (RAG == PRF)          : {metrics.gain_loss.neutral_count:,} ({metrics.gain_loss.neutral_pct:.2f}%)")
    print(f"    - Degradadas (RAG < PRF)          : {metrics.gain_loss.degraded_count:,} ({metrics.gain_loss.degraded_pct:.2f}%)")
    print(f"    - Balance Neto                    : {metrics.gain_loss.net_wins:+d} consultas")
    print()
    print("  Tiempos de Ejecución:")
    print(f"    - ⭐ TIEMPO PROMEDIO INFERENCIA LLM : {metrics.mean_inference_time_seconds:.4f} s")
    print(f"    - Tiempo Total Promedio RAG        : {metrics.timings_rag.total_time_mean:.4f} s")
    print(f"    - Tiempo Total Promedio PRF        : {metrics.timings_prf.total_time_mean:.4f} s")
    print(f"    - Text Fetch (RAG)                 : {metrics.timings_rag.text_fetch_mean*1000:.2f} ms ({metrics.timings_rag.text_fetch_pct:.2f}% marginal)")
    print()
    if metrics.lexicon.has_term_data:
        print("  Filtrado de Lexicón / Alucinación del LLM:")
        print(f"    - Términos propuestos por LLM      : {metrics.lexicon.total_proposed:,} (media: {metrics.lexicon.n_terms_proposed_mean:.2f})")
        print(f"    - Aceptados en lexicón             : {metrics.lexicon.total_kept:,} ({metrics.lexicon.survival_rate_pct:.2f}%)")
        print(f"    - Descartados (Alucinación)        : {metrics.lexicon.total_discarded:,} ({metrics.lexicon.hallucination_rate_pct:.2f}%)")
    print("─" * 70)

    print("\n📁 Gráficos y reportes generados:")
    for key, path in fig_paths.items():
        print(f"  • [{key}] → {path}")

    print("\n✅ Análisis completado con éxito.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
