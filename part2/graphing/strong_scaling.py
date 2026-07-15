import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANKS = [1, 2, 4]

GHOST_MODES = {
    "alltoallv": {
        "time_col": "ghost_exchange_alltoallv_time_mean_max",
        "suffix": "mpi",
        "label": "K+MPI",
        "note": "MPI_Alltoallv only",
    },
    "fullghost": {
        "time_col": "ghost_exchange_time_mean_max",
        "suffix": "fullghost",
        "label": "K+G",
        "note": "full ghost phase",
    },
}


def clean_name(path_value: str) -> str:
    name = os.path.basename(str(path_value).strip())
    if name.endswith(".mtx"):
        name = name[:-4]
    return name


def safe_filename(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("-", "_")
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
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["matrix"] = df["matrix"].astype(str).str.strip()
    df["matrix_short"] = df["matrix"].apply(clean_name)
    df["result_type"] = df["result_type"].astype(str).str.strip()
    df["nprocs"] = pd.to_numeric(df["nprocs"], errors="raise").astype(int)

    for col in ["kernel_time_mean_max", mode["time_col"]]:
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

    raise ValueError(f"Unknown dataset filter: {dataset}")


def prepare_times(
    df: pd.DataFrame,
    backend: str,
    ghost_mode: str,
) -> pd.DataFrame:
    mode = GHOST_MODES[ghost_mode]

    df = df[df["result_type"] == backend].copy()

    if df.empty:
        raise ValueError(f"No rows found for backend '{backend}'")

    df["kernel_time"] = df["kernel_time_mean_max"]
    df["step_time"] = df["kernel_time_mean_max"] + df[mode["time_col"]]

    grouped = (
        df.groupby(["matrix_short", "nprocs"], as_index=False)
        .agg(
            kernel_time=("kernel_time", "mean"),
            step_time=("step_time", "mean"),
        )
    )

    return grouped


def compute_speedup_tables(times: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kernel_pivot = times.pivot(
        index="matrix_short", columns="nprocs", values="kernel_time"
    )
    step_pivot = times.pivot(
        index="matrix_short", columns="nprocs", values="step_time"
    )

    missing_kernel = [r for r in RANKS if r not in kernel_pivot.columns]
    missing_step = [r for r in RANKS if r not in step_pivot.columns]

    if missing_kernel or missing_step:
        raise ValueError(
            f"Missing rank counts. Kernel missing: {missing_kernel}; "
            f"step missing: {missing_step}"
        )

    kernel_pivot = kernel_pivot.dropna(subset=RANKS)
    step_pivot = step_pivot.dropna(subset=RANKS)

    common_matrices = kernel_pivot.index.intersection(step_pivot.index)

    if len(common_matrices) == 0:
        raise ValueError("No common matrices found across all requested ranks")

    kernel_pivot = kernel_pivot.loc[common_matrices]
    step_pivot = step_pivot.loc[common_matrices]

    per_matrix_rows = []

    for matrix in common_matrices:
        row = {"matrix": matrix}

        for rank in RANKS:
            row[f"kernel_time_p{rank}"] = kernel_pivot.loc[matrix, rank]
            row[f"step_time_p{rank}"] = step_pivot.loc[matrix, rank]

            row[f"kernel_speedup_p{rank}"] = (
                kernel_pivot.loc[matrix, 1] / kernel_pivot.loc[matrix, rank]
            )
            row[f"step_speedup_p{rank}"] = (
                step_pivot.loc[matrix, 1] / step_pivot.loc[matrix, rank]
            )

            row[f"kernel_efficiency_p{rank}"] = (
                row[f"kernel_speedup_p{rank}"] / rank
            )
            row[f"step_efficiency_p{rank}"] = (
                row[f"step_speedup_p{rank}"] / rank
            )

        per_matrix_rows.append(row)

    per_matrix = pd.DataFrame(per_matrix_rows)

    summary_rows = []

    for rank in RANKS:
        kernel_speedup = per_matrix[f"kernel_speedup_p{rank}"]
        step_speedup = per_matrix[f"step_speedup_p{rank}"]
        kernel_eff = per_matrix[f"kernel_efficiency_p{rank}"]
        step_eff = per_matrix[f"step_efficiency_p{rank}"]

        summary_rows.append(
            {
                "nprocs": rank,
                "kernel_geomean_speedup": geometric_mean(kernel_speedup),
                "step_geomean_speedup": geometric_mean(step_speedup),
                "kernel_geomean_efficiency": geometric_mean(kernel_eff),
                "step_geomean_efficiency": geometric_mean(step_eff),
                "kernel_mean_speedup": float(kernel_speedup.mean()),
                "step_mean_speedup": float(step_speedup.mean()),
                "kernel_mean_efficiency": float(kernel_eff.mean()),
                "step_mean_efficiency": float(step_eff.mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return per_matrix, summary


def plot_summary(
    summary: pd.DataFrame,
    backend: str,
    dataset: str,
    ghost_mode: str,
    output_dir: Path,
) -> Path:
    mode = GHOST_MODES[ghost_mode]

    ranks = summary["nprocs"].to_numpy()
    kernel_speedup = summary["kernel_geomean_speedup"].to_numpy()
    step_speedup = summary["step_geomean_speedup"].to_numpy()

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    ax.plot(
        ranks,
        ranks,
        linestyle="--",
        linewidth=1.0,
        marker=None,
        label="Ideal",
    )

    ax.plot(
        ranks,
        kernel_speedup,
        marker="o",
        linewidth=1.4,
        label="Kernel",
    )

    ax.plot(
        ranks,
        step_speedup,
        marker="s",
        linewidth=1.4,
        label=mode["label"],
    )

    ax.set_xticks(ranks)
    ax.set_xlabel("MPI ranks / GPUs")
    ax.set_ylabel("Geomean speedup")
    ax.set_title(f"{backend} strong scaling")

    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    ymax = max(
        4.2,
        np.nanmax(kernel_speedup) * 1.18,
        np.nanmax(step_speedup) * 1.18,
    )
    ax.set_ylim(bottom=0.0, top=ymax)

    ax.legend(loc="upper left", frameon=True)

    output_path = (
        output_dir
        / f"strong_scaling_{safe_filename(backend)}_{dataset}_{mode['suffix']}.pdf"
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return output_path


def plot_per_matrix_debug(
    per_matrix: pd.DataFrame,
    backend: str,
    dataset: str,
    ghost_mode: str,
    output_dir: Path,
) -> Path:
    mode = GHOST_MODES[ghost_mode]

    matrices = per_matrix["matrix"].tolist()
    labels = [m.replace("_", r"\_") for m in matrices]
    x = np.arange(len(matrices))

    fig_width = max(7.2, 0.72 * len(matrices))
    fig, ax = plt.subplots(figsize=(fig_width, 3.4))

    values = [
        ("Kernel, 4 ranks", per_matrix["kernel_speedup_p4"].to_numpy()),
        (f"{mode['label']}, 4 ranks", per_matrix["step_speedup_p4"].to_numpy()),
    ]

    total_group_width = 0.72
    gap = 0.04
    bar_width = (total_group_width - gap) / len(values)
    start_offset = -total_group_width / 2 + bar_width / 2

    for i, (label, series) in enumerate(values):
        offset = start_offset + i * (bar_width + gap)

        bars = ax.bar(
            x + offset,
            series,
            width=bar_width,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            label_type="edge",
            fontsize=6.5,
            padding=2,
            rotation=90,
        )

    ax.axhline(1.0, linestyle="-", linewidth=0.8)
    ax.axhline(4.0, linestyle=":", linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("4-rank speedup")
    ax.set_xlabel("Matrix")
    ax.set_title(f"{backend} 4-rank speedup by matrix")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.margins(x=0.01)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(4.4, ymax * 1.22))

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=2,
        frameon=True,
    )

    output_path = (
        output_dir
        / f"strong_scaling_{safe_filename(backend)}_{dataset}_{mode['suffix']}_debug_per_matrix.pdf"
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a strong-scaling plot comparing kernel-only speedup "
            "against a selected communication-inclusive timing."
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
        help="Output directory for figures and derived CSV files.",
    )
    parser.add_argument(
        "--backend",
        default="CSR_GPU",
        help="Backend/result_type to plot. Default: CSR_GPU.",
    )
    parser.add_argument(
        "--dataset",
        choices=["real", "synthetic", "all"],
        default="real",
        help="Dataset subset to plot. Default: real.",
    )
    parser.add_argument(
        "--ghost-mode",
        choices=sorted(GHOST_MODES.keys()),
        default="alltoallv",
        help=(
            "Ghost timing to add to kernel time. "
            "alltoallv uses only MPI_Alltoallv time; "
            "fullghost uses the full ghost-vector phase. Default: alltoallv."
        ),
    )
    parser.add_argument(
        "--debug-per-matrix",
        action="store_true",
        help="Also write a wide per-matrix diagnostic plot.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = GHOST_MODES[args.ghost_mode]
    stem = f"strong_scaling_{safe_filename(args.backend)}_{args.dataset}_{mode['suffix']}"

    df = load_compute_csvs(args.compute, args.ghost_mode)
    df = filter_dataset(df, args.dataset)

    times = prepare_times(df, args.backend, args.ghost_mode)
    per_matrix, summary = compute_speedup_tables(times)

    per_matrix_csv = output_dir / f"{stem}_per_matrix.csv"
    summary_csv = output_dir / f"{stem}_summary.csv"

    per_matrix.to_csv(per_matrix_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    written = [
        per_matrix_csv,
        summary_csv,
        plot_summary(summary, args.backend, args.dataset, args.ghost_mode, output_dir),
    ]

    if args.debug_per_matrix:
        written.append(
            plot_per_matrix_debug(
                per_matrix,
                args.backend,
                args.dataset,
                args.ghost_mode,
                output_dir,
            )
        )

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
