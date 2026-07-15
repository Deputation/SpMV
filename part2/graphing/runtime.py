import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GHOST_MODES = {
    "alltoallv": {
        "time_col": "ghost_exchange_alltoallv_time_mean_max",
        "suffix": "mpi",
        "label": "MPI exchange",
        "note": "MPI_Alltoallv only",
    },
    "fullghost": {
        "time_col": "ghost_exchange_time_mean_max",
        "suffix": "fullghost",
        "label": "Full ghost exchange",
        "note": "full ghost exchange",
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

    raise ValueError(f"Unknown dataset: {dataset}")


def prepare_breakdown(
    df: pd.DataFrame,
    backend: str,
    rank: int,
    ghost_mode: str,
    sort_by: str,
) -> tuple[pd.DataFrame, str]:
    mode = GHOST_MODES[ghost_mode]
    ghost_col = mode["time_col"]

    df = df[(df["result_type"] == backend) & (df["nprocs"] == rank)].copy()

    if df.empty:
        raise ValueError(f"No rows found for backend '{backend}' with nprocs={rank}")

    input_order = list(dict.fromkeys(df["matrix_short"].tolist()))
    order_map = {name: i for i, name in enumerate(input_order)}

    agg_dict = {
        "kernel_time": ("kernel_time_mean_max", "mean"),
        "ghost_time": (ghost_col, "mean"),
    }

    optional_cols = [
        "rank_flops_max_over_avg",
        "ghost_values_received_avg",
        "ghost_bytes_received_avg",
        "ghost_bytes_received_max",
    ]

    for col in optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg_dict[col] = (col, "mean")

    out = df.groupby("matrix_short", as_index=False).agg(**agg_dict)

    out["total_time"] = out["kernel_time"] + out["ghost_time"]
    out["ghost_fraction"] = np.where(
        out["total_time"] > 0.0,
        out["ghost_time"] / out["total_time"],
        np.nan,
    )
    out["ghost_mode"] = ghost_mode
    out["input_order"] = out["matrix_short"].map(order_map)

    if sort_by == "ghost-fraction":
        out = out.sort_values(["ghost_fraction", "total_time"], ascending=[False, False])
    elif sort_by == "total-time":
        out = out.sort_values("total_time", ascending=False)
    elif sort_by == "matrix":
        out = out.sort_values("matrix_short")
    elif sort_by == "input":
        out = out.sort_values("input_order")
    else:
        raise ValueError(f"Unknown sort mode: {sort_by}")

    out = out.drop(columns=["input_order"])
    return out.reset_index(drop=True), mode["label"]


def plot_breakdown(
    breakdown: pd.DataFrame,
    backend: str,
    dataset: str,
    rank: int,
    ghost_mode: str,
    ghost_label: str,
    output_dir: Path,
) -> Path:
    mode = GHOST_MODES[ghost_mode]

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

    plot_df = breakdown.copy()

    plot_df["kernel_ms"] = plot_df["kernel_time"] * 1000.0
    plot_df["ghost_ms"] = plot_df["ghost_time"] * 1000.0
    plot_df["total_ms"] = plot_df["total_time"] * 1000.0

    y = np.arange(len(plot_df))

    fig_height = max(2.8, 0.30 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(3.45, fig_height))

    ax.barh(
        y,
        plot_df["kernel_ms"],
        edgecolor="black",
        linewidth=0.45,
        label="Kernel",
    )

    ax.barh(
        y,
        plot_df["ghost_ms"],
        left=plot_df["kernel_ms"],
        edgecolor="black",
        linewidth=0.45,
        label=ghost_label,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["matrix_short"])
    ax.invert_yaxis()

    ax.set_xlabel("Time per SpMV step (ms)")
    ax.set_title(f"{backend} runtime breakdown, {rank} GPUs")

    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    xmax = plot_df["total_ms"].max()
    ax.set_xlim(left=0.0, right=xmax * 1.23)

    for i, row in plot_df.iterrows():
        ghost_percent = row["ghost_fraction"] * 100.0
        ax.text(
            row["total_ms"] + xmax * 0.025,
            i,
            f"{ghost_percent:.0f}%",
            va="center",
            ha="left",
            fontsize=6.5,
        )

    lgd = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=True,
        borderpad=0.35,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    output_path = (
        output_dir
        / f"runtime_breakdown_{safe_filename(backend)}_{dataset}_p{rank}_{mode['suffix']}.pdf"
    )

    fig.savefig(
        output_path,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a runtime breakdown figure showing local kernel time "
            "and ghost-exchange time for a selected backend and rank count."
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
        "--rank",
        type=int,
        default=4,
        help="MPI rank/GPU count to plot. Default: 4.",
    )
    parser.add_argument(
        "--ghost-mode",
        choices=sorted(GHOST_MODES.keys()),
        default="alltoallv",
        help=(
            "Ghost timing to stack with kernel time. "
            "alltoallv uses MPI_Alltoallv time only; "
            "fullghost uses the full ghost exchange time. Default: alltoallv."
        ),
    )
    parser.add_argument(
        "--sort",
        choices=["ghost-fraction", "total-time", "matrix", "input"],
        default="ghost-fraction",
        help="Bar ordering. Default: ghost-fraction.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = GHOST_MODES[args.ghost_mode]
    stem = (
        f"runtime_breakdown_{safe_filename(args.backend)}_"
        f"{args.dataset}_p{args.rank}_{mode['suffix']}"
    )

    df = load_compute_csvs(args.compute, args.ghost_mode)
    df = filter_dataset(df, args.dataset)

    breakdown, ghost_label = prepare_breakdown(
        df=df,
        backend=args.backend,
        rank=args.rank,
        ghost_mode=args.ghost_mode,
        sort_by=args.sort,
    )

    csv_path = output_dir / f"{stem}.csv"
    breakdown.to_csv(csv_path, index=False)

    fig_path = plot_breakdown(
        breakdown=breakdown,
        backend=args.backend,
        dataset=args.dataset,
        rank=args.rank,
        ghost_mode=args.ghost_mode,
        ghost_label=ghost_label,
        output_dir=output_dir,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
