#!/bin/bash

output="baseline_cpu_merged.csv"

mapfile -t files < <(find . -maxdepth 1 -type f -name "*.csv" ! -name "$output" | sort)

if (( ${#files[@]} == 0 )); then
  echo "no cpu baseline csv files"
  exit 0
fi

head -n 1 "${files[0]}" > "$output"

for file in "${files[@]}"; do
  tail -n +2 "$file" >> "$output"
done

echo "merged ${#files[@]} files into $output"
