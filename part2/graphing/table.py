import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


RANKS = [1, 2, 4]
DEFAULT_BACKENDS = ["CSR_GPU", "CSR_CUSPARSE"]


def clean_name(path_value: str) -> str:
    name = os.path.basename(str(path_value).strip())
    if name.endswith(".mtx"):
        name = name[:-4]
    return name


def latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def geometric_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    values = values[values > 0.0]

    if len(values) == 0:
        return np.nan

    return float(np.exp(np.mean(np.log(values))))


def load_compute_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    required_cols = [
        "matrix",
        "result_type",
        "nprocs",
        "kernel_time_mean_max",
        "ghost_exchange_alltoallv_time_mean_max",
        "ghost_exchange_time_mean_max",
        "global_kernel_gflops",
        "global_kernel_plus_alltoallv_ghost_gflops",
        "global_kernel_plus_full_ghost_gflops",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["matrix"] = df["matrix"].astype(str).str.strip()
    df["matrix_short"] = df["matrix"].apply(clean_name)
    df["result_type"] = df["result_type"].astype(str).str.strip()
    df["nprocs"] = pd.to_numeric(df["nprocs"], errors="raise").astype(int)

    numeric_cols = [
        "kernel_time_mean_max",
        "ghost_exchange_alltoallv_time_mean_max",
        "ghost_exchange_time_mean_max",
        "global_kernel_gflops",
        "global_kernel_plus_alltoallv_ghost_gflops",
        "global_kernel_plus_full_ghost_gflops",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def filter_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "all":
        return df.copy()

    is_synthetic = df["matrix"].str.contains("synthetic", case=False, na=False)

    if dataset == "real":
        return df[~is_synthetic].copy()

    if dataset == "synthetic":
        return df[is_synthetic].copy()

    raise ValueError(f"Unknown dataset: {dataset}")


def prepare_grouped(df: pd.DataFrame, backends: list[str]) -> pd.DataFrame:
    df = df[df["result_type"].isin(backends)].copy()

    if df.empty:
        raise ValueError(f"No rows found for requested backends: {backends}")

    missing_backends = sorted(set(backends) - set(df["result_type"].unique()))
    if missing_backends:
        raise ValueError(f"No rows found for backend(s): {missing_backends}")

    df["kernel_time_ms"] = df["kernel_time_mean_max"] * 1000.0
    df["mpi_time_ms"] = (
        df["kernel_time_mean_max"]
        + df["ghost_exchange_alltoallv_time_mean_max"]
    ) * 1000.0
    df["ghost_time_ms"] = (
        df["kernel_time_mean_max"]
        + df["ghost_exchange_time_mean_max"]
    ) * 1000.0

    grouped = (
        df.groupby(["result_type", "matrix_short", "nprocs"], as_index=False)
        .agg(
            kernel_time_ms=("kernel_time_ms", "mean"),
            mpi_time_ms=("mpi_time_ms", "mean"),
            ghost_time_ms=("ghost_time_ms", "mean"),
            kernel_gflops=("global_kernel_gflops", "mean"),
            mpi_gflops=("global_kernel_plus_alltoallv_ghost_gflops", "mean"),
            ghost_gflops=("global_kernel_plus_full_ghost_gflops", "mean"),
        )
    )

    return grouped


def common_matrices_for_backend(bdf: pd.DataFrame, backend: str) -> list[str]:
    sets = []

    for rank in RANKS:
        part = bdf[bdf["nprocs"] == rank]

        if part.empty:
            raise ValueError(f"Backend {backend} is missing rank {rank}")

        sets.append(set(part["matrix_short"]))

    common = set.intersection(*sets)

    if len(common) == 0:
        raise ValueError(f"No common matrices across ranks for backend {backend}")

    return sorted(common)


def compute_summary(grouped: pd.DataFrame, backends: list[str]) -> pd.DataFrame:
    rows = []

    for backend in backends:
        bdf = grouped[grouped["result_type"] == backend].copy()

        missing_ranks = sorted(set(RANKS) - set(bdf["nprocs"].unique()))
        if missing_ranks:
            raise ValueError(f"Backend {backend} is missing rank(s): {missing_ranks}")

        common = common_matrices_for_backend(bdf, backend)
        bdf = bdf[bdf["matrix_short"].isin(common)].copy()

        base = bdf[bdf["nprocs"] == 1][
            ["matrix_short", "kernel_time_ms", "mpi_time_ms", "ghost_time_ms"]
        ].rename(
            columns={
                "kernel_time_ms": "kernel_time_ms_p1",
                "mpi_time_ms": "mpi_time_ms_p1",
                "ghost_time_ms": "ghost_time_ms_p1",
            }
        )

        merged = bdf.merge(base, on="matrix_short", how="inner")

        for rank in RANKS:
            part = merged[merged["nprocs"] == rank].copy()

            if part.empty:
                raise ValueError(f"No common matrices for {backend} at rank {rank}")

            part["kernel_speedup"] = (
                part["kernel_time_ms_p1"] / part["kernel_time_ms"]
            )
            part["mpi_speedup"] = part["mpi_time_ms_p1"] / part["mpi_time_ms"]
            part["ghost_speedup"] = (
                part["ghost_time_ms_p1"] / part["ghost_time_ms"]
            )

            part["mpi_efficiency"] = part["mpi_speedup"] / rank
            part["ghost_efficiency"] = part["ghost_speedup"] / rank

            rows.append(
                {
                    "backend": backend,
                    "nprocs": rank,
                    "matrix_count": int(len(part)),
                    "kernel_time_ms_geomean": geometric_mean(part["kernel_time_ms"]),
                    "mpi_time_ms_geomean": geometric_mean(part["mpi_time_ms"]),
                    "ghost_time_ms_geomean": geometric_mean(part["ghost_time_ms"]),
                    "kernel_gflops_geomean": geometric_mean(part["kernel_gflops"]),
                    "mpi_gflops_geomean": geometric_mean(part["mpi_gflops"]),
                    "ghost_gflops_geomean": geometric_mean(part["ghost_gflops"]),
                    "kernel_speedup_geomean": geometric_mean(part["kernel_speedup"]),
                    "mpi_speedup_geomean": geometric_mean(part["mpi_speedup"]),
                    "ghost_speedup_geomean": geometric_mean(part["ghost_speedup"]),
                    "mpi_efficiency_geomean": geometric_mean(part["mpi_efficiency"]),
                    "ghost_efficiency_geomean": geometric_mean(part["ghost_efficiency"]),
                }
            )

    return pd.DataFrame(rows)


def fmt_time(value: float) -> str:
    return f"{value:.3f}"


def fmt_value(value: float) -> str:
    return f"{value:.2f}"


def backend_latex(name: str) -> str:
    return f"\\texttt{{{latex_escape(name)}}}"


def write_latex_table(
    summary: pd.DataFrame,
    output_path: Path,
    dataset: str,
) -> None:
    rows = []

    for _, row in summary.iterrows():
        rows.append(
            "            "
            f"{backend_latex(row['backend'])} & "
            f"{int(row['nprocs'])} & "
            f"{fmt_time(row['kernel_time_ms_geomean'])} & "
            f"{fmt_time(row['mpi_time_ms_geomean'])} & "
            f"{fmt_time(row['ghost_time_ms_geomean'])} & "
            f"{fmt_value(row['mpi_gflops_geomean'])} & "
            f"{fmt_value(row['ghost_gflops_geomean'])} & "
            f"{fmt_value(row['kernel_speedup_geomean'])} & "
            f"{fmt_value(row['mpi_speedup_geomean'])} & "
            f"{fmt_value(row['ghost_speedup_geomean'])} & "
            f"{fmt_value(row['mpi_efficiency_geomean'])} & "
            f"{fmt_value(row['ghost_efficiency_geomean'])} \\\\"
        )

    content = rf"""\begin{{table}}[!t]
    \renewcommand{{\arraystretch}}{{1.10}}
    \centering
    \caption{{Geometric-mean CSR strong-scaling results.}}
    \label{{tab:csr_results_summary}}
    \resizebox{{\columnwidth}}{{!}}{{
        \begin{{tabular}}{{l r r r r r r r r r r r}}
            \hline
            \textbf{{Backend}} & \textbf{{P}} & \textbf{{$T_K$}} & \textbf{{$T_M$}} & \textbf{{$T_G$}} & \textbf{{$F_M$}} & \textbf{{$F_G$}} & \textbf{{$S_K$}} & \textbf{{$S_M$}} & \textbf{{$S_G$}} & \textbf{{$E_M$}} & \textbf{{$E_G$}} \\
            \hline
"""
    content += "\n".join(rows)
    content += r"""
            \hline
        \end{tabular}
    }

    \vspace{0.4ex}
    \footnotesize
    $T$ reports time, $F$ throughput, $S$ speedup, and $E$ parallel efficiency. $P$ is the number of MPI ranks/GPUs. Times are in ms and throughput is in GF/s.
    Subscripts denote: $K$, kernel only; $M$, kernel plus ghost-vector \texttt{MPI\_Alltoallv}; $G$, kernel plus full ghost exchange.
    Speedup is relative to one GPU for the same backend, and efficiency is $E=S/P$.
\end{table}
"""

    output_path.write_text(content)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table summarizing CSR strong-scaling metrics "
            "with kernel, MPI-exchange, and full-ghost-exchange timings."
        )
    )
    parser.add_argument(
        "--compute",
        nargs="+",
        required=True,
        help="Compute CSV files, e.g. 1_compute.csv 2_compute.csv 4_compute.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the .tex table and summary CSV.",
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        default=DEFAULT_BACKENDS,
        help="Backends/result_type values to include. Default: CSR_GPU CSR_CUSPARSE.",
    )
    parser.add_argument(
        "--dataset",
        choices=["real", "synthetic", "all"],
        default="real",
        help="Dataset subset to summarize. Default: real.",
    )
    parser.add_argument(
        "--table-name",
        default="csr_results_summary_table.tex",
        help="Output LaTeX table filename. Default: csr_results_summary_table.tex.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_compute_csvs(args.compute)
    df = filter_dataset(df, args.dataset)
    grouped = prepare_grouped(df, args.backend)
    summary = compute_summary(grouped, args.backend)

    stem = Path(args.table_name).stem
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / args.table_name

    summary.to_csv(csv_path, index=False)
    write_latex_table(summary, tex_path, args.dataset)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
