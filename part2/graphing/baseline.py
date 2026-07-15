import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BACKENDS = ["CSR_GPU", "CSR_CUSPARSE"]


GHOST_MODES = {
    "alltoallv": {
        "time_col": "ghost_exchange_alltoallv_time_mean_max",
        "gflops_col": "global_kernel_plus_alltoallv_ghost_gflops",
        "suffix": "mpi",
        "timing_label": "Kernel+MPI",
        "note": r"\textit{Kernel+MPI} adds ghost-vector \texttt{MPI\_Alltoallv} time.",
    },
    "fullghost": {
        "time_col": "ghost_exchange_time_mean_max",
        "gflops_col": "global_kernel_plus_full_ghost_gflops",
        "suffix": "fullghost",
        "timing_label": "Kernel+ghost",
        "note": r"\textit{Kernel+ghost} adds the full measured ghost exchange.",
    },
}


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


def load_compute_csvs(paths: list[str], ghost_mode: str) -> pd.DataFrame:
    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    mode = GHOST_MODES[ghost_mode]

    required_cols = [
        "matrix",
        "result_type",
        "nprocs",
        "kernel_time_mean_max",
        mode["time_col"],
        "global_kernel_gflops",
        mode["gflops_col"],
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Compute CSV is missing required column: {col}")

    df["matrix"] = df["matrix"].astype(str).str.strip()
    df["matrix_short"] = df["matrix"].apply(clean_name)
    df["result_type"] = df["result_type"].astype(str).str.strip()
    df["nprocs"] = pd.to_numeric(df["nprocs"], errors="raise").astype(int)

    numeric_cols = [
        "kernel_time_mean_max",
        mode["time_col"],
        "global_kernel_gflops",
        mode["gflops_col"],
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def load_baseline_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = [
        "matrix",
        "spmv_mean_s",
        "gflops",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Baseline CSV is missing required column: {col}")

    df["matrix"] = df["matrix"].astype(str).str.strip()
    df["matrix_short"] = df["matrix"].apply(clean_name)
    df["spmv_mean_s"] = pd.to_numeric(df["spmv_mean_s"], errors="raise")
    df["gflops"] = pd.to_numeric(df["gflops"], errors="raise")

    return (
        df.groupby("matrix_short", as_index=False)
        .agg(
            cpu_time_s=("spmv_mean_s", "mean"),
            cpu_gflops=("gflops", "mean"),
        )
    )


def filter_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "all":
        return df.copy()

    if "matrix" in df.columns:
        source = df["matrix"]
    else:
        source = df["matrix_short"]

    is_synthetic = source.astype(str).str.contains("synthetic", case=False, na=False)

    if dataset == "real":
        return df[~is_synthetic].copy()

    if dataset == "synthetic":
        return df[is_synthetic].copy()

    raise ValueError(f"Unknown dataset: {dataset}")


def extract_gpu_method(
    compute_df: pd.DataFrame,
    backend: str,
    rank: int,
    timing: str,
    ghost_mode: str,
) -> pd.DataFrame:
    mode = GHOST_MODES[ghost_mode]

    df = compute_df[
        (compute_df["result_type"] == backend)
        & (compute_df["nprocs"] == rank)
    ].copy()

    if df.empty:
        raise ValueError(f"No compute rows found for backend={backend}, rank={rank}")

    if timing == "kernel":
        df["time_s"] = df["kernel_time_mean_max"]
        df["gflops"] = df["global_kernel_gflops"]
        timing_label = "Kernel"
    elif timing == "step":
        df["time_s"] = df["kernel_time_mean_max"] + df[mode["time_col"]]
        df["gflops"] = df[mode["gflops_col"]]
        timing_label = mode["timing_label"]
    else:
        raise ValueError(f"Unknown timing: {timing}")

    grouped = (
        df.groupby("matrix_short", as_index=False)
        .agg(
            time_s=("time_s", "mean"),
            gflops=("gflops", "mean"),
        )
    )

    grouped["implementation"] = backend
    grouped["resources"] = f"{rank} GPU" if rank == 1 else f"{rank} GPUs"
    grouped["timing"] = timing_label

    return grouped


def build_method_frames(
    baseline_df: pd.DataFrame,
    compute_df: pd.DataFrame,
    backends: list[str],
    ghost_mode: str,
) -> list[pd.DataFrame]:
    frames = []

    cpu = baseline_df.copy()
    cpu = cpu.rename(
        columns={
            "cpu_time_s": "time_s",
            "cpu_gflops": "gflops",
        }
    )
    cpu["implementation"] = "Provided CPU CSR"
    cpu["resources"] = "1 CPU process"
    cpu["timing"] = "Kernel"

    frames.append(
        cpu[
            [
                "matrix_short",
                "implementation",
                "resources",
                "timing",
                "time_s",
                "gflops",
            ]
        ]
    )

    for backend in backends:
        frames.append(
            extract_gpu_method(
                compute_df=compute_df,
                backend=backend,
                rank=1,
                timing="kernel",
                ghost_mode=ghost_mode,
            )
        )
        frames.append(
            extract_gpu_method(
                compute_df=compute_df,
                backend=backend,
                rank=4,
                timing="step",
                ghost_mode=ghost_mode,
            )
        )

    return frames


def common_matrices(frames: list[pd.DataFrame]) -> list[str]:
    common = set(frames[0]["matrix_short"])

    for frame in frames[1:]:
        common &= set(frame["matrix_short"])

    if len(common) == 0:
        raise ValueError("No common matrices found across baseline and GPU results")

    return sorted(common)


def summarize_methods(frames: list[pd.DataFrame], common: list[str]) -> pd.DataFrame:
    cpu_frame = frames[0]
    cpu_time = (
        cpu_frame[cpu_frame["matrix_short"].isin(common)]
        .set_index("matrix_short")["time_s"]
    )

    rows = []

    for frame in frames:
        part = frame[frame["matrix_short"].isin(common)].copy()
        part = part.set_index("matrix_short").loc[common].reset_index()

        speedups = (
            cpu_time.loc[part["matrix_short"]].to_numpy()
            / part["time_s"].to_numpy()
        )

        rows.append(
            {
                "implementation": part["implementation"].iloc[0],
                "resources": part["resources"].iloc[0],
                "timing": part["timing"].iloc[0],
                "matrix_count": len(part),
                "time_ms_geomean": geometric_mean(part["time_s"] * 1000.0),
                "gflops_geomean": geometric_mean(part["gflops"]),
                "speedup_vs_cpu_geomean": geometric_mean(pd.Series(speedups)),
            }
        )

    return pd.DataFrame(rows)


def format_impl(name: str) -> str:
    if name.startswith("CSR_"):
        return f"\\texttt{{{latex_escape(name)}}}"
    return latex_escape(name)


def dataset_label(dataset: str) -> str:
    if dataset == "real":
        return "real matrices"
    if dataset == "synthetic":
        return "synthetic matrices"
    return "matrices"


def write_latex_table(
    summary: pd.DataFrame,
    output_path: Path,
    dataset: str,
    ghost_mode: str,
) -> None:
    mode = GHOST_MODES[ghost_mode]

    rows = []

    for _, row in summary.iterrows():
        rows.append(
            "            "
            f"{format_impl(row['implementation'])} & "
            f"{latex_escape(row['resources'])} & "
            f"{latex_escape(row['timing'])} & "
            f"{row['time_ms_geomean']:.3f} & "
            f"{row['gflops_geomean']:.2f} & "
            f"{row['speedup_vs_cpu_geomean']:.2f} \\\\"
        )

    data_label = dataset_label(dataset)

    content = rf"""\begin{{table}}[!t]
    \renewcommand{{\arraystretch}}{{1.12}}
    \centering
    \caption{{Comparison with the provided CPU CSR baseline.}}
    \label{{tab:baseline_comparison_{mode["suffix"]}}}
    \resizebox{{\columnwidth}}{{!}}{{
        \begin{{tabular}}{{l l l r r r}}
            \hline
            \textbf{{Implementation}} & \textbf{{Resources}} & \textbf{{Timing}} & \textbf{{Time ms}} & \textbf{{GF/s}} & \textbf{{Speedup}} \\
            \hline
"""
    content += "\n".join(rows)
    content += rf"""
            \hline
        \end{{tabular}}
    }}

    \vspace{{0.4ex}}
    \footnotesize
    Geometric means across the {data_label}. \textit{{Kernel}} is SpMV kernel time.
    {mode["note"]} Speedup is relative to the provided serial CPU CSR baseline.
\end{{table}}
"""

    output_path.write_text(content)


def default_table_name(ghost_mode: str) -> str:
    return f"baseline_comparison_{GHOST_MODES[ghost_mode]['suffix']}_table.tex"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact LaTeX table comparing the provided CPU CSR baseline "
            "against selected GPU distributed SpMV measurements."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Merged baseline CPU CSV, e.g. baseline_cpu_merged.csv",
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
        help="Backends to include. Default: CSR_GPU CSR_CUSPARSE.",
    )
    parser.add_argument(
        "--dataset",
        choices=["real", "synthetic", "all"],
        default="real",
        help="Dataset subset to summarize. Default: real.",
    )
    parser.add_argument(
        "--ghost-mode",
        choices=sorted(GHOST_MODES.keys()),
        default="alltoallv",
        help=(
            "Ghost timing added to the four-GPU rows. "
            "alltoallv uses only MPI_Alltoallv time; "
            "fullghost uses the full ghost exchange time. Default: alltoallv."
        ),
    )
    parser.add_argument(
        "--table-name",
        default=None,
        help="Output LaTeX table filename. Default depends on --ghost-mode.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_name = args.table_name or default_table_name(args.ghost_mode)

    baseline_df = load_baseline_csv(args.baseline)
    compute_df = load_compute_csvs(args.compute, args.ghost_mode)

    baseline_df = filter_dataset(baseline_df, args.dataset)
    compute_df = filter_dataset(compute_df, args.dataset)

    frames = build_method_frames(
        baseline_df=baseline_df,
        compute_df=compute_df,
        backends=args.backend,
        ghost_mode=args.ghost_mode,
    )

    common = common_matrices(frames)
    summary = summarize_methods(frames, common)

    stem = Path(table_name).stem
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / table_name

    summary.to_csv(csv_path, index=False)
    write_latex_table(summary, tex_path, args.dataset, args.ghost_mode)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Common matrices used: {len(common)}")


if __name__ == "__main__":
    main()
