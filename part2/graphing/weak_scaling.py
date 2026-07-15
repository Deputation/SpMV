import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANK_TO_SIZE = {
    1: "10M",
    2: "20M",
    4: "40M",
}

RANKS = [1, 2, 4]

GHOST_MODES = {
    "alltoallv": {
        "time_col": "ghost_exchange_alltoallv_time_mean_max",
        "suffix": "mpi",
        "label": "K+MPI",
    },
    "fullghost": {
        "time_col": "ghost_exchange_time_mean_max",
        "suffix": "fullghost",
        "label": "K+G",
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


def extract_size_label(matrix_name: str) -> str:
    name = clean_name(matrix_name)

    match = re.search(r"(\d+)M", name, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}M"

    raise ValueError(f"Could not extract synthetic size label from matrix name: {name}")


def extract_profile(matrix_name: str) -> str:
    name = clean_name(matrix_name).lower()

    if "unbalanced" in name:
        return "Unbalanced"

    return "Balanced"


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


def prepare_weak_scaling_rows(
    df: pd.DataFrame,
    backend: str,
    ghost_mode: str,
) -> pd.DataFrame:
    mode = GHOST_MODES[ghost_mode]
    ghost_col = mode["time_col"]

    df = df[df["result_type"] == backend].copy()

    if df.empty:
        raise ValueError(f"No rows found for backend '{backend}'")

    df = df[df["matrix"].str.contains("synthetic", case=False, na=False)].copy()

    if df.empty:
        raise ValueError("No synthetic matrices found in the compute CSV files")

    df["size_label"] = df["matrix_short"].apply(extract_size_label)
    df["profile"] = df["matrix_short"].apply(extract_profile)

    df["kernel_time"] = df["kernel_time_mean_max"]
    df["ghost_time"] = df[ghost_col]
    df["step_time"] = df["kernel_time"] + df["ghost_time"]

    df["expected_size"] = df["nprocs"].map(RANK_TO_SIZE)
    df = df[df["size_label"] == df["expected_size"]].copy()

    if df.empty:
        raise ValueError(
            "No rows matched the weak-scaling rule: "
            "1 rank -> 10M, 2 ranks -> 20M, 4 ranks -> 40M"
        )

    grouped = (
        df.groupby(["profile", "nprocs", "size_label"], as_index=False)
        .agg(
            kernel_time=("kernel_time", "mean"),
            ghost_time=("ghost_time", "mean"),
            step_time=("step_time", "mean"),
        )
    )

    required_profiles = ["Balanced", "Unbalanced"]

    missing = []
    for profile in required_profiles:
        profile_df = grouped[grouped["profile"] == profile]
        present_ranks = set(profile_df["nprocs"].tolist())

        for rank in RANKS:
            if rank not in present_ranks:
                missing.append(f"{profile}, {rank} rank(s)")

    if missing:
        raise ValueError("Missing weak-scaling points: " + ", ".join(missing))

    rows = []

    for profile in required_profiles:
        profile_df = grouped[grouped["profile"] == profile].copy()
        profile_df = profile_df.set_index("nprocs")

        kernel_base = profile_df.loc[1, "kernel_time"]
        step_base = profile_df.loc[1, "step_time"]

        for rank in RANKS:
            kernel_time = profile_df.loc[rank, "kernel_time"]
            ghost_time = profile_df.loc[rank, "ghost_time"]
            step_time = profile_df.loc[rank, "step_time"]

            rows.append(
                {
                    "profile": profile,
                    "nprocs": rank,
                    "size_label": RANK_TO_SIZE[rank],
                    "kernel_time": kernel_time,
                    "ghost_time": ghost_time,
                    "step_time": step_time,
                    "kernel_normalized_time": kernel_time / kernel_base,
                    "step_normalized_time": step_time / step_base,
                    "kernel_weak_scaling_efficiency": kernel_base / kernel_time,
                    "step_weak_scaling_efficiency": step_base / step_time,
                    "ghost_fraction_step": (
                        ghost_time / step_time if step_time > 0.0 else np.nan
                    ),
                    "ghost_mode": ghost_mode,
                }
            )

    return pd.DataFrame(rows)


def plot_weak_scaling(
    weak: pd.DataFrame,
    backend: str,
    ghost_mode: str,
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

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    ranks = np.array(RANKS)

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="Ideal",
    )

    style = {
        ("Balanced", "kernel_normalized_time"): ("o", "-", "Bal., K"),
        ("Balanced", "step_normalized_time"): ("s", "-", f"Bal., {mode['label']}"),
        ("Unbalanced", "kernel_normalized_time"): ("o", ":", "Unbal., K"),
        ("Unbalanced", "step_normalized_time"): ("s", ":", f"Unbal., {mode['label']}"),
    }

    for (profile, col), (marker, linestyle, label) in style.items():
        part = weak[weak["profile"] == profile].copy()
        part = part.set_index("nprocs").loc[RANKS]

        ax.plot(
            ranks,
            part[col].to_numpy(),
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            label=label,
        )

    ax.set_xticks(ranks)
    ax.set_xticklabels([f"{r} GPU\n{RANK_TO_SIZE[r]} NNZ" for r in ranks])

    ax.set_xlabel("Weak-scaling configuration")
    ax.set_ylabel("Normalized time")
    ax.set_title(f"{backend} weak scaling")

    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    ymax = max(
        1.2,
        weak["kernel_normalized_time"].max(),
        weak["step_normalized_time"].max(),
    )
    ax.set_ylim(bottom=0.0, top=ymax * 1.15)

    lgd = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=True,
        ncol=2,
        borderpad=0.35,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    output_path = (
        output_dir
        / f"weak_scaling_{safe_filename(backend)}_{mode['suffix']}.pdf"
    )

    fig.savefig(
        output_path,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    return output_path


def plot_raw_time_debug(
    weak: pd.DataFrame,
    backend: str,
    ghost_mode: str,
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

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    ranks = np.array(RANKS)

    for profile, linestyle, short_name in [
        ("Balanced", "-", "Bal."),
        ("Unbalanced", ":", "Unbal."),
    ]:
        part = weak[weak["profile"] == profile].copy()
        part = part.set_index("nprocs").loc[RANKS]

        ax.plot(
            ranks,
            part["kernel_time"].to_numpy() * 1000.0,
            marker="o",
            linestyle=linestyle,
            linewidth=1.35,
            label=f"{short_name} K",
        )

        ax.plot(
            ranks,
            part["step_time"].to_numpy() * 1000.0,
            marker="s",
            linestyle=linestyle,
            linewidth=1.35,
            label=f"{short_name} {mode['label']}",
        )

    ax.set_xticks(ranks)
    ax.set_xticklabels([f"{r} GPU\n{RANK_TO_SIZE[r]} NNZ" for r in ranks])

    ax.set_xlabel("Weak-scaling configuration")
    ax.set_ylabel("Time per SpMV step (ms)")
    ax.set_title(f"{backend} weak scaling")

    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    lgd = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=True,
        ncol=2,
        borderpad=0.35,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    output_path = (
        output_dir
        / f"weak_scaling_{safe_filename(backend)}_{mode['suffix']}_raw_time_debug.pdf"
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
            "Generate weak-scaling plots for synthetic matrices. "
            "Uses 10M/1 rank, 20M/2 ranks, and 40M/4 ranks."
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
        "--raw-time-debug",
        action="store_true",
        help="Also write a raw-time diagnostic plot.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_compute_csvs(args.compute, args.ghost_mode)
    weak = prepare_weak_scaling_rows(df, args.backend, args.ghost_mode)

    stem = f"weak_scaling_{safe_filename(args.backend)}_{GHOST_MODES[args.ghost_mode]['suffix']}"
    weak_csv = output_dir / f"{stem}.csv"
    weak.to_csv(weak_csv, index=False)

    written = [
        weak_csv,
        plot_weak_scaling(weak, args.backend, args.ghost_mode, output_dir),
    ]

    if args.raw_time_debug:
        written.append(
            plot_raw_time_debug(weak, args.backend, args.ghost_mode, output_dir)
        )

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
