#!/bin/bash

head -n 1 $(ls results/*.csv | head -n 1) > results.csv
tail -q -n +2 results/*.csv >> results.csv
