#!/bin/bash

mkdir -p merged_results

for nproc in 1 2 4; do
  for kind in compute parsing; do
    output="merged_results/${nproc}_${kind}.csv"

    mapfile -t files < <(find results -maxdepth 1 -type f -name "${nproc}_*_${kind}.csv" | sort)

    if (( ${#files[@]} == 0 )); then
      echo "no ${nproc}_${kind}"
      continue
    fi

    head -n 1 "${files[0]}" > "$output"

    for file in "${files[@]}"; do
      tail -n +2 "$file" >> "$output"
    done
  done
done
