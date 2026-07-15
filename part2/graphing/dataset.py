import argparse
import os
from pathlib import Path

import pandas as pd


def latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def compact_matrix_name(path_value: str) -> str:
    name = os.path.basename(str(path_value))
    if name.endswith(".mtx"):
        name = name[:-4]
    return latex_escape(name)


def yn(value) -> str:
    return "Y" if str(value).strip().lower() == "true" else "N"


def load_info(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "sparsity",
        "max to avg",
        "coeff variation",
        "rows",
        "cols",
        "nnz",
        "total elements",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")

    df["avg nnz row"] = df["nnz"] / df["rows"]
    return df


def write_real_table(df: pd.DataFrame, output_dir: str) -> None:
    rows = []

    for _, row in df.iterrows():
        matrix = compact_matrix_name(row["matrix_name"])
        nrows = f"{int(row['rows'])}"
        ncols = f"{int(row['cols'])}"
        nnz_m = f"{row['nnz'] / 1e6:.2f}"
        avg_nnz_row = f"{row['avg nnz row']:.2f}"
        max_avg = f"{row['max to avg']:.2f}"
        cv = f"{row['coeff variation']:.2f}"
        sym = yn(row["symmetric"])
        pattern = yn(row["graph"])

        rows.append(
            f"            {matrix} & {nrows} & {ncols} & {nnz_m} & "
            f"{avg_nnz_row} & {max_avg} & {cv} & {sym} & {pattern} \\\\"
        )

    content = r"""\begin{table}[!t]
    \renewcommand{\arraystretch}{1.15}
    \centering
    \caption{Real matrix dataset characteristics.}
    \label{tab:real_dataset}
    \resizebox{\columnwidth}{!}{
        \begin{tabular}{l r r r r r r c c}
            \hline
            \textbf{Matrix} & \textbf{Rows} & \textbf{Cols} & \textbf{NNZ (M)} & \textbf{Avg.} & \textbf{Max/Avg} & \textbf{CV} & \textbf{Sym.} & \textbf{Pat.} \\
            \hline
"""
    content += "\n".join(rows)
    content += r"""
            \hline
        \end{tabular}
    }
\end{table}
"""

    Path(output_dir, "real_dataset_table.tex").write_text(content)


def synthetic_profile(matrix_name: str) -> str:
    name = os.path.basename(str(matrix_name)).lower()
    return "Unbal." if "unbalanced" in name else "Bal."


def synthetic_size_label(matrix_name: str, nnz: float) -> str:
    name = os.path.basename(str(matrix_name)).replace(".mtx", "")

    for token in ["10M", "20M", "40M"]:
        if token.lower() in name.lower():
            return token

    return f"{nnz / 1e6:.0f}M"


def write_synthetic_table(df: pd.DataFrame, output_dir: str) -> None:
    rows = []

    for _, row in df.iterrows():
        size = synthetic_size_label(row["matrix_name"], row["nnz"])
        profile = synthetic_profile(row["matrix_name"])
        nnz_m = f"{row['nnz'] / 1e6:.2f}"
        avg_nnz_row = f"{row['avg nnz row']:.2f}"
        max_avg = f"{row['max to avg']:.2f}"
        cv = f"{row['coeff variation']:.2f}"

        rows.append(
            f"            {size} & {profile} & {nnz_m} & "
            f"{avg_nnz_row} & {max_avg} & {cv} \\\\"
        )

    content = r"""\begin{table}[!t]
    \renewcommand{\arraystretch}{1.15}
    \centering
    \caption{Synthetic matrix dataset characteristics.}
    \label{tab:synthetic_dataset}
    \resizebox{\columnwidth}{!}{
        \begin{tabular}{l c r r r r}
            \hline
            \textbf{Size} & \textbf{Profile} & \textbf{NNZ (M)} & \textbf{Avg.} & \textbf{Max/Avg} & \textbf{CV} \\
            \hline
"""
    content += "\n".join(rows)
    content += r"""
            \hline
        \end{tabular}
    }
\end{table}
"""

    Path(output_dir, "synthetic_dataset_table.tex").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX dataset tables for the SpMV paper."
    )
    parser.add_argument(
        "--real",
        required=True,
        help="Path to real_matrices_info.csv",
    )
    parser.add_argument(
        "--synthetic",
        required=True,
        help="Path to synthetic_matrices_info.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the .tex tables will be written",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    real_df = load_info(args.real)
    synthetic_df = load_info(args.synthetic)

    write_real_table(real_df, args.output)
    write_synthetic_table(synthetic_df, args.output)

    print(f"Wrote {Path(args.output, 'real_dataset_table.tex')}")
    print(f"Wrote {Path(args.output, 'synthetic_dataset_table.tex')}")


if __name__ == "__main__":
    main()
