import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RANKS = [2, 4]


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
        "ghost_exchange_alltoallv_time_mean_max",
        "ghost_exchange_time_mean_max",
        "ghost_values_received_avg",
        "ghost_values_received_max",
        "ghost_bytes_received_avg",
        "ghost_bytes_received_max",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Compute CSV is missing required column: {col}")

    df["matrix"] = df["matrix"].astype(str).str.strip()
    df["matrix_short"] = df["matrix"].apply(clean_name)
    df["result_type"] = df["result_type"].astype(str).str.strip()
    df["nprocs"] = pd.to_numeric(df["nprocs"], errors="raise").astype(int)

    numeric_cols = [
        "ghost_exchange_alltoallv_time_mean_max",
        "ghost_exchange_time_mean_max",
        "ghost_values_received_avg",
        "ghost_values_received_max",
        "ghost_bytes_received_avg",
        "ghost_bytes_received_max",
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


def dataset_label(dataset: str) -> str:
    if dataset == "real":
        return "real matrices"
    if dataset == "synthetic":
        return "synthetic matrices"
    return "matrices"


def extract_backend(
    df: pd.DataFrame,
    backend: str,
    ranks: list[int],
    role: str,
) -> pd.DataFrame:
    out = df[
        (df["result_type"] == backend)
        & (df["nprocs"].isin(ranks))
    ].copy()

    if out.empty:
        raise ValueError(f"No rows found for backend={backend}")

    out = (
        out.groupby(["matrix_short", "nprocs"], as_index=False)
        .agg(
            mpi_time_s=("ghost_exchange_alltoallv_time_mean_max", "mean"),
            full_time_s=("ghost_exchange_time_mean_max", "mean"),
            ghost_values_avg=("ghost_values_received_avg", "mean"),
            ghost_values_max=("ghost_values_received_max", "mean"),
            ghost_bytes_avg=("ghost_bytes_received_avg", "mean"),
            ghost_bytes_max=("ghost_bytes_received_max", "mean"),
        )
    )

    out = out.rename(
        columns={
            "mpi_time_s": f"{role}_mpi_time_s",
            "full_time_s": f"{role}_full_time_s",
            "ghost_values_avg": f"{role}_ghost_values_avg",
            "ghost_values_max": f"{role}_ghost_values_max",
            "ghost_bytes_avg": f"{role}_ghost_bytes_avg",
            "ghost_bytes_max": f"{role}_ghost_bytes_max",
        }
    )

    return out


def build_comparison(
    df: pd.DataFrame,
    host_backend: str,
    device_backend: str,
    ranks: list[int],
) -> pd.DataFrame:
    host = extract_backend(df, host_backend, ranks, "host")
    device = extract_backend(df, device_backend, ranks, "device")

    merged = host.merge(device, on=["matrix_short", "nprocs"], how="inner")

    if merged.empty:
        raise ValueError("No common matrices found between host and device exchange backends")

    merged["mpi_speedup_host_over_device"] = (
        merged["host_mpi_time_s"] / merged["device_mpi_time_s"]
    )
    merged["full_speedup_host_over_device"] = (
        merged["host_full_time_s"] / merged["device_full_time_s"]
    )

    merged["ghost_values_avg"] = merged["device_ghost_values_avg"]
    merged["ghost_values_max"] = merged["device_ghost_values_max"]
    merged["ghost_comm_mb_avg"] = merged["device_ghost_bytes_avg"] / 1e6
    merged["ghost_comm_mb_max"] = merged["device_ghost_bytes_max"] / 1e6

    return merged


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rank in sorted(comparison["nprocs"].unique()):
        part = comparison[comparison["nprocs"] == rank].copy()

        rows.append(
            {
                "nprocs": int(rank),
                "matrix_count": int(len(part)),
                "ghost_values_avg_k_geomean": geometric_mean(part["ghost_values_avg"] / 1e3),
                "ghost_comm_mb_avg_geomean": geometric_mean(part["ghost_comm_mb_avg"]),
                "host_mpi_ms_geomean": geometric_mean(part["host_mpi_time_s"] * 1000.0),
                "device_mpi_ms_geomean": geometric_mean(part["device_mpi_time_s"] * 1000.0),
                "mpi_speedup_geomean": geometric_mean(part["mpi_speedup_host_over_device"]),
                "host_full_ms_geomean": geometric_mean(part["host_full_time_s"] * 1000.0),
                "device_full_ms_geomean": geometric_mean(part["device_full_time_s"] * 1000.0),
                "full_speedup_geomean": geometric_mean(part["full_speedup_host_over_device"]),
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
    host_backend: str,
    device_backend: str,
) -> None:
    rows = []

    for _, row in summary.iterrows():
        rows.append(
            "            "
            f"{int(row['nprocs'])} & "
            f"{fmt_value(row['ghost_values_avg_k_geomean'])} & "
            f"{fmt_value(row['ghost_comm_mb_avg_geomean'])} & "
            f"{fmt_time(row['host_mpi_ms_geomean'])} & "
            f"{fmt_time(row['device_mpi_ms_geomean'])} & "
            f"{fmt_value(row['mpi_speedup_geomean'])} & "
            f"{fmt_time(row['host_full_ms_geomean'])} & "
            f"{fmt_time(row['device_full_ms_geomean'])} & "
            f"{fmt_value(row['full_speedup_geomean'])} \\\\"
        )

    content = rf"""\begin{{table}}[!t]
    \renewcommand{{\arraystretch}}{{1.12}}
    \centering
    \caption{{Host and CUDA-aware ghost exchange comparison.}}
    \label{{tab:cuda_aware_exchange}}
    \resizebox{{\columnwidth}}{{!}}{{
        \begin{{tabular}}{{r r r r r r r r r}}
            \hline
            \textbf{{P}} & \textbf{{Ghost K}} & \textbf{{MB}} & \textbf{{$H_M$}} & \textbf{{$D_M$}} & \textbf{{$S_M$}} & \textbf{{$H_G$}} & \textbf{{$D_G$}} & \textbf{{$S_G$}} \\
            \hline
"""
    content += "\n".join(rows)
    content += rf"""
            \hline
        \end{{tabular}}
    }}

    \vspace{{0.4ex}}
    \footnotesize
    Geometric means on the {dataset_label(dataset)}. Times are in ms.
    $H$ uses {backend_latex(host_backend)}, $D$ uses {backend_latex(device_backend)}.
    $M$ is \texttt{{MPI\_Alltoallv}} time; $G$ is full ghost exchange. $S=H/D$.
\end{{table}}
"""

    output_path.write_text(content)


def default_table_name() -> str:
    return "cuda_aware_exchange_table.tex"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare host-resident and CUDA-aware/device-resident ghost exchange timings."
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
        help="Output directory for the .tex table and summary CSV files.",
    )
    parser.add_argument(
        "--host-backend",
        default="CSR_CPU",
        help="Backend using host-resident ghost exchange. Default: CSR_CPU.",
    )
    parser.add_argument(
        "--device-backend",
        default="CSR_GPU",
        help="Backend using CUDA-aware/device-resident ghost exchange. Default: CSR_GPU.",
    )
    parser.add_argument(
        "--dataset",
        choices=["real", "synthetic", "all"],
        default="real",
        help="Dataset subset to summarize. Default: real.",
    )
    parser.add_argument(
        "--rank",
        nargs="+",
        type=int,
        default=DEFAULT_RANKS,
        help="MPI rank/GPU counts to include. Default: 2 4.",
    )
    parser.add_argument(
        "--table-name",
        default=default_table_name(),
        help="Output LaTeX table filename. Default: cuda_aware_exchange_table.tex.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_compute_csvs(args.compute)
    df = filter_dataset(df, args.dataset)

    comparison = build_comparison(
        df=df,
        host_backend=args.host_backend,
        device_backend=args.device_backend,
        ranks=args.rank,
    )
    summary = summarize(comparison)

    stem = Path(args.table_name).stem
    per_matrix_csv = output_dir / f"{stem}_per_matrix.csv"
    summary_csv = output_dir / f"{stem}.csv"
    tex_path = output_dir / args.table_name

    comparison.to_csv(per_matrix_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    write_latex_table(
        summary=summary,
        output_path=tex_path,
        dataset=args.dataset,
        host_backend=args.host_backend,
        device_backend=args.device_backend,
    )

    print(f"Wrote {per_matrix_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
