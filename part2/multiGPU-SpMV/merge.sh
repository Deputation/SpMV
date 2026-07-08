awk 'FNR==1 && NR!=1 { next } { print }' baseline_cpu_*.csv > baseline_cpu_merged.csv
