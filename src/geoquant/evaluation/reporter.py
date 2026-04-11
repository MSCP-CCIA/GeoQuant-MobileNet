"""
Reporter: serializa resultados de evaluación a JSON, CSV y LaTeX.
"""

import csv
import json
from pathlib import Path
from typing import Any


class Reporter:
    """
    Serializa el dict de resultados de EvaluationSuite a múltiples formatos.

    Args:
        output_dir: Directorio donde se guardan los reportes.
    """

    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, results: dict, filename: str = "results.json") -> Path:
        out = self.output_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return out

    # ------------------------------------------------------------------
    # CSV (aplanado bloque→métrica→valor)
    # ------------------------------------------------------------------

    def to_csv(self, results: dict, filename: str = "results.csv") -> Path:
        out = self.output_dir / filename
        rows = []
        for block, metrics in results.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    rows.append({"block": block, "metric": metric, "value": value})
            else:
                rows.append({"block": "global", "metric": block, "value": metrics})

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["block", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        return out

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def to_latex(self, results: dict, filename: str = "results.tex", caption: str = "Resultados de Evaluación Geométrica") -> Path:
        out = self.output_dir / filename
        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            r"  \begin{tabular}{llr}",
            r"    \toprule",
            r"    \textbf{Bloque} & \textbf{Métrica} & \textbf{Valor} \\",
            r"    \midrule",
        ]
        for block, metrics in results.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    lines.append(f"    {block} & {metric} & {val_str} \\\\")
            else:
                val_str = f"{metrics:.4f}" if isinstance(metrics, float) else str(metrics)
                lines.append(f"    global & {block} & {val_str} \\\\")
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Reporte completo (todos los formatos)
    # ------------------------------------------------------------------

    def save_all(self, results: dict, stem: str = "results") -> dict:
        """Guarda JSON, CSV y LaTeX. Retorna dict con rutas."""
        return {
            "json": str(self.to_json(results, f"{stem}.json")),
            "csv": str(self.to_csv(results, f"{stem}.csv")),
            "latex": str(self.to_latex(results, f"{stem}.tex")),
        }