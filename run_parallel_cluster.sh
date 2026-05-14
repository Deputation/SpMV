#!/bin/bash

MATRICES=(
    "dataset/FullChip/FullChip.mtx"
    "dataset/ASIC_680ks/ASIC_680ks.mtx"
    "dataset/Rucci1/Rucci1.mtx"
    "dataset/eu-2005/eu-2005.mtx"
    "dataset/ldoor/ldoor.mtx"
    "dataset/rajat31/rajat31.mtx"
    "dataset/Ga41As41H72/Ga41As41H72.mtx"
    "dataset/Si41Ge41H72/Si41Ge41H72.mtx"
    "dataset/webbase-1M/webbase-1M.mtx"
    "dataset/bone010/bone010.mtx"
    # "dataset/bone010/bone010_M.mtx"
)

CACHE_MATRICES=(
    "dataset/Ga41As41H72/Ga41As41H72.mtx"
)

for MAT in "${CACHE_MATRICES[@]}"; do
    NAME=$(basename "$MAT" .mtx)

    sbatch --partition=edu-short \
           --account=gpu.computing26 \
           --nodes=1 \
           --ntasks=1 \
           --gres=gpu:1 \
           --cpus-per-task=1 \
           --time=00:05:00 \
           --nodelist=edu01 \
           --job-name="cache_coo" \
           --output="logs/cache_coo_${NAME}-%j.out" \
           --error="logs/cache_coo_${NAME}-%j.err" <<EOF
#!/bin/bash
module load CUDA/12.5.0
valgrind --tool=cachegrind --cache-sim=yes \
         --log-file="logs/cache_summary_coo_${NAME}_%p.txt" \
         --cachegrind-out-file="logs/cache_profile_coo_${NAME}_%p.out" \
         ./output/spmv_cpu_coo "${MAT}"
EOF

    sbatch --partition=edu-short \
           --account=gpu.computing26 \
           --nodes=1 \
           --ntasks=1 \
           --gres=gpu:1 \
           --cpus-per-task=1 \
           --time=00:05:00 \
           --nodelist=edu01 \
           --job-name="cache_csr" \
           --output="logs/cache_csr_${NAME}-%j.out" \
           --error="logs/cache_csr_${NAME}-%j.err" <<EOF
#!/bin/bash
module load CUDA/12.5.0
valgrind --tool=cachegrind --cache-sim=yes \
         --log-file="logs/cache_summary_csr_${NAME}_%p.txt" \
         --cachegrind-out-file="logs/cache_profile_csr_${NAME}_%p.out" \
         ./output/spmv_cpu_csr "${MAT}"
EOF

    echo "Queued COO and CSR cache jobs for: $NAME"
done

for MAT in "${MATRICES[@]}"; do
    NAME=$(basename "$MAT" .mtx)

    sbatch --partition=edu-short \
           --account=gpu.computing26 \
           --nodes=1 \
           --ntasks=1 \
           --gres=gpu:1 \
           --cpus-per-task=12 \
           --time=00:05:00 \
           --nodelist=edu01 \
           --job-name="bench_coo" \
           --output="logs/coo_${NAME}-%j.out" \
           --error="logs/coo_${NAME}-%j.err" <<EOF
#!/bin/bash
module load CUDA/12.5.0
./output/spmv_coo ${MAT}
EOF

    sbatch --partition=edu-short \
           --account=gpu.computing26 \
           --nodes=1 \
           --ntasks=1 \
           --gres=gpu:1 \
           --cpus-per-task=12 \
           --time=00:05:00 \
           --nodelist=edu01 \
           --job-name="bench_csr" \
           --output="logs/csr_${NAME}-%j.out" \
           --error="logs/csr_${NAME}-%j.err" <<EOF
#!/bin/bash
module load CUDA/12.5.0
./output/spmv_csr ${MAT}
EOF

    echo "Queued COO and CSR standard jobs for: $NAME"
done

echo "All jobs submitted!"
