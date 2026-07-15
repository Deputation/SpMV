import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_BYTES = 8
VALUE_BYTES = 4


GHOST_MODES = {
    "alltoallv": {
        "time_col": "ghost_exchange_alltoallv_time_mean_max",
        "suffix": "mpi",
        "share_note": r"\texttt{MPI\_Alltoallv} ghost exchange",
    },
    "fullghost": {
        "time_col": "ghost_exchange_time_mean_max",
        "suffix": "fullghost",
        "share_note": "full ghost exchange",
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


def find_matrix_column(df: pd.DataFrame) -> str:
    for candidate in ["matrix", "matrix_name"]:
        if candidate in df.columns:
            return candidate

    raise ValueError("Could not find matrix column. Expected 'matrix' or 'matrix_name'.")


def load_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    matrix_col = find_matrix_column(df)
    df["matrix"] = df[matrix_col].astype(str).str.strip()
    df["matrix_short"] = df["matrix"].apply(clean_name)

    if "nprocs" in df.columns:
        df["nprocs"] = pd.to_numeric(df["nprocs"], errors="raise").astype(int)

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
    return "all matrices"


def prepare_matrix_info(info_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["rows", "cols"]

    for col in required_cols:
        if col not in info_df.columns:
            raise ValueError(f"Matrix-info CSV is missing required column: {col}")
        info_df[col] = pd.to_numeric(info_df[col], errors="raise")

    return (
        info_df.groupby("matrix_short", as_index=False)
        .agg(
            rows=("rows", "mean"),
            cols=("cols", "mean"),
        )
    )


def prepare_compute(
    compute_df: pd.DataFrame,
    backend: str,
    rank: int,
    ghost_mode: str,
) -> pd.DataFrame:
    mode = GHOST_MODES[ghost_mode]

    required_cols = [
        "result_type",
        "nprocs",
        "kernel_time_mean_max",
        mode["time_col"],
        "ghost_values_received_avg",
        "ghost_values_received_max",
        "ghost_bytes_received_avg",
        "ghost_bytes_received_max",
    ]

    for col in required_cols:
        if col not in compute_df.columns:
            raise ValueError(f"Compute CSV is missing required column: {col}")

    compute_df = compute_df.copy()
    compute_df["result_type"] = compute_df["result_type"].astype(str).str.strip()
    compute_df["nprocs"] = pd.to_numeric(
        compute_df["nprocs"], errors="raise"
    ).astype(int)

    numeric_cols = [
        "kernel_time_mean_max",
        mode["time_col"],
        "ghost_values_received_avg",
        "ghost_values_received_max",
        "ghost_bytes_received_avg",
        "ghost_bytes_received_max",
    ]

    for col in numeric_cols:
        compute_df[col] = pd.to_numeric(compute_df[col], errors="raise")

    out = compute_df[
        (compute_df["result_type"] == backend)
        & (compute_df["nprocs"] == rank)
    ].copy()

    if out.empty:
        raise ValueError(f"No compute rows found for backend={backend}, nprocs={rank}")

    out["ghost_time"] = out[mode["time_col"]]
    out["step_time"] = out["kernel_time_mean_max"] + out["ghost_time"]
    out["ghost_share"] = np.where(
        out["step_time"] > 0.0,
        out["ghost_time"] / out["step_time"],
        np.nan,
    )

    grouped = (
        out.groupby("matrix_short", as_index=False)
        .agg(
            kernel_time=("kernel_time_mean_max", "mean"),
            ghost_time=("ghost_time", "mean"),
            step_time=("step_time", "mean"),
            ghost_share=("ghost_share", "mean"),
            ghost_values_avg=("ghost_values_received_avg", "mean"),
            ghost_values_max=("ghost_values_received_max", "mean"),
            ghost_bytes_avg=("ghost_bytes_received_avg", "mean"),
            ghost_bytes_max=("ghost_bytes_received_max", "mean"),
        )
    )

    return grouped


def prepare_parsing(
    parsing_df: pd.DataFrame,
    rank: int,
) -> pd.DataFrame:
    required_cols = [
        "nprocs",
        "rank_nnz_min",
        "rank_nnz_avg",
        "rank_nnz_max",
        "rank_nnz_max_over_avg",
    ]

    for col in required_cols:
        if col not in parsing_df.columns:
            raise ValueError(f"Parsing CSV is missing required column: {col}")

    parsing_df = parsing_df.copy()
    parsing_df["nprocs"] = pd.to_numeric(
        parsing_df["nprocs"], errors="raise"
    ).astype(int)

    numeric_cols = [
        "rank_nnz_min",
        "rank_nnz_avg",
        "rank_nnz_max",
        "rank_nnz_max_over_avg",
    ]

    for col in numeric_cols:
        parsing_df[col] = pd.to_numeric(parsing_df[col], errors="raise")

    out = parsing_df[parsing_df["nprocs"] == rank].copy()

    if out.empty:
        raise ValueError(f"No parsing rows found for nprocs={rank}")

    grouped = (
        out.groupby("matrix_short", as_index=False)
        .agg(
            rank_nnz_min=("rank_nnz_min", "mean"),
            rank_nnz_avg=("rank_nnz_avg", "mean"),
            rank_nnz_max=("rank_nnz_max", "mean"),
            rank_nnz_max_over_avg=("rank_nnz_max_over_avg", "mean"),
        )
    )

    return grouped


def add_memory_estimate(df: pd.DataFrame, rank: int) -> pd.DataFrame:
    df = df.copy()

    row_avg = df["rows"] / rank
    row_max = np.ceil(df["rows"] / rank)

    col_avg = df["cols"] / rank
    col_max = np.ceil(df["cols"] / rank)

    mem_avg_bytes = (
        (row_avg + 1.0) * INDEX_BYTES
        + df["rank_nnz_avg"] * (INDEX_BYTES + VALUE_BYTES)
        + row_avg * VALUE_BYTES
        + (col_avg + df["ghost_values_avg"]) * VALUE_BYTES
    )

    mem_max_bytes = (
        (row_max + 1.0) * INDEX_BYTES
        + df["rank_nnz_max"] * (INDEX_BYTES + VALUE_BYTES)
        + row_max * VALUE_BYTES
        + (col_max + df["ghost_values_max"]) * VALUE_BYTES
    )

    df["mem_mb_avg"] = mem_avg_bytes / 1e6
    df["mem_mb_max"] = mem_max_bytes / 1e6

    return df


def build_diagnostics_table(
    compute_df: pd.DataFrame,
    parsing_df: pd.DataFrame,
    info_df: pd.DataFrame,
    backend: str,
    dataset: str,
    rank: int,
    ghost_mode: str,
    sort_by: str,
) -> pd.DataFrame:
    compute_df = filter_dataset(compute_df, dataset)
    parsing_df = filter_dataset(parsing_df, dataset)
    info_df = filter_dataset(info_df, dataset)

    compute = prepare_compute(compute_df, backend, rank, ghost_mode)
    parsing = prepare_parsing(parsing_df, rank)
    info = prepare_matrix_info(info_df)

    merged = parsing.merge(compute, on="matrix_short", how="inner")
    merged = merged.merge(info, on="matrix_short", how="inner")

    if merged.empty:
        raise ValueError(
            "No common matrices found between compute, parsing, and matrix-info CSVs"
        )

    merged = add_memory_estimate(merged, rank)

    merged["rank_nnz_min_m"] = merged["rank_nnz_min"] / 1e6
    merged["rank_nnz_avg_m"] = merged["rank_nnz_avg"] / 1e6
    merged["rank_nnz_max_m"] = merged["rank_nnz_max"] / 1e6

    merged["ghost_values_avg_k"] = merged["ghost_values_avg"] / 1e3
    merged["ghost_values_max_k"] = merged["ghost_values_max"] / 1e3
    merged["ghost_comm_mb_avg"] = merged["ghost_bytes_avg"] / 1e6
    merged["ghost_comm_mb_max"] = merged["ghost_bytes_max"] / 1e6
    merged["ghost_share_percent"] = merged["ghost_share"] * 100.0
    merged["ghost_mode"] = ghost_mode

    if sort_by == "ghost-share":
        merged = merged.sort_values(
            ["ghost_share", "ghost_comm_mb_avg"],
            ascending=[False, False],
        )
    elif sort_by == "ghost-mb":
        merged = merged.sort_values("ghost_comm_mb_avg", ascending=False)
    elif sort_by == "nnz-balance":
        merged = merged.sort_values("rank_nnz_max_over_avg", ascending=False)
    elif sort_by == "memory":
        merged = merged.sort_values("mem_mb_max", ascending=False)
    elif sort_by == "matrix":
        merged = merged.sort_values("matrix_short")
    else:
        raise ValueError(f"Unknown sort mode: {sort_by}")

    return merged.reset_index(drop=True)


def fmt_nnz(min_value: float, avg: float, max_value: float) -> str:
    return f"{min_value:.3f}/{avg:.3f}/{max_value:.3f}"


def fmt_kvalues(avg: float, max_value: float) -> str:
    return f"{avg:.1f}/{max_value:.1f}"


def fmt_mb(avg: float, max_value: float) -> str:
    return f"{avg:.2f}/{max_value:.2f}"


def fmt_percent(value: float) -> str:
    return f"{value:.0f}\\%"


def write_latex_table(
    diagnostics: pd.DataFrame,
    output_path: Path,
    backend: str,
    dataset: str,
    rank: int,
    ghost_mode: str,
) -> None:
    mode = GHOST_MODES[ghost_mode]

    rows = []

    for _, row in diagnostics.iterrows():
        rows.append(
            "            "
            f"{latex_escape(row['matrix_short'])} & "
            f"{fmt_nnz(row['rank_nnz_min_m'], row['rank_nnz_avg_m'], row['rank_nnz_max_m'])} & "
            f"{fmt_kvalues(row['ghost_values_avg_k'], row['ghost_values_max_k'])} & "
            f"{fmt_mb(row['ghost_comm_mb_avg'], row['ghost_comm_mb_max'])} & "
            f"{fmt_mb(row['mem_mb_avg'], row['mem_mb_max'])} & "
            f"{fmt_percent(row['ghost_share_percent'])} \\\\"
        )

    gpu_label = "GPU" if rank == 1 else "GPUs"

    content = rf"""\begin{{table}}[!t]
    \renewcommand{{\arraystretch}}{{1.12}}
    \centering
    \caption{{Partition and communication diagnostics on {rank} {gpu_label}.}}
    \label{{tab:communication_diagnostics_{mode["suffix"]}}}
    \resizebox{{\columnwidth}}{{!}}{{
        \begin{{tabular}}{{l r r r r r}}
            \hline
            \textbf{{Matrix}} & \textbf{{NNZ M min/avg/max}} & \textbf{{Ghost vals. K}} & \textbf{{Ghost recv. MB}} & \textbf{{Mem. MB}} & \textbf{{Ghost share}} \\
            \hline
"""
    content += "\n".join(rows)
    content += rf"""
            \hline
        \end{{tabular}}
    }}

    \vspace{{0.4ex}}
    \footnotesize
    \texttt{{{latex_escape(backend)}}}, {dataset_label(dataset)}.
    NNZ is min/avg/max; other values are average/max per rank.
    Ghost share is measured against kernel plus {mode["share_note"]}.
    Memory is the steady-state CSR estimate.
\end{{table}}
"""

    output_path.write_text(content)


def default_table_name(ghost_mode: str) -> str:
    return f"communication_diagnostics_{GHOST_MODES[ghost_mode]['suffix']}_table.tex"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact LaTeX table reporting per-rank NNZ balance, "
            "ghost communication volume, and estimated steady-state memory footprint."
        )
    )
    parser.add_argument(
        "--compute",
        nargs="+",
        required=True,
        help="Compute CSV files, e.g. 1_compute.csv 2_compute.csv 4_compute.csv",
    )
    parser.add_argument(
        "--parsing",
        nargs="+",
        required=True,
        help="Parsing CSV files, e.g. 1_parsing.csv 2_parsing.csv 4_parsing.csv",
    )
    parser.add_argument(
        "--matrix-info",
        nargs="+",
        required=True,
        help="Matrix-info CSV files, e.g. real_matrices_info.csv synthetic_matrices_info.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the .tex table and summary CSV.",
    )
    parser.add_argument(
        "--backend",
        default="CSR_GPU",
        help="Backend/result_type to summarize. Default: CSR_GPU.",
    )
    parser.add_argument(
        "--dataset",
        choices=["real", "synthetic", "all"],
        default="real",
        help="Dataset subset to summarize. Default: real.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="MPI rank/GPU count to summarize. Default: 4.",
    )
    parser.add_argument(
        "--ghost-mode",
        choices=sorted(GHOST_MODES.keys()),
        default="alltoallv",
        help=(
            "Ghost timing used for the ghost-share column. "
            "alltoallv uses only MPI_Alltoallv time; "
            "fullghost uses the full ghost exchange time. Default: alltoallv."
        ),
    )
    parser.add_argument(
        "--sort",
        choices=["ghost-share", "ghost-mb", "nnz-balance", "memory", "matrix"],
        default="ghost-share",
        help="Row ordering. Default: ghost-share.",
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

    compute_df = load_csvs(args.compute)
    parsing_df = load_csvs(args.parsing)
    info_df = load_csvs(args.matrix_info)

    diagnostics = build_diagnostics_table(
        compute_df=compute_df,
        parsing_df=parsing_df,
        info_df=info_df,
        backend=args.backend,
        dataset=args.dataset,
        rank=args.rank,
        ghost_mode=args.ghost_mode,
        sort_by=args.sort,
    )

    stem = Path(table_name).stem
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / table_name

    diagnostics.to_csv(csv_path, index=False)
    write_latex_table(
        diagnostics=diagnostics,
        output_path=tex_path,
        backend=args.backend,
        dataset=args.dataset,
        rank=args.rank,
        ghost_mode=args.ghost_mode,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
