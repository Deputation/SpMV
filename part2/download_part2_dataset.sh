#!/bin/bash

mkdir -p dataset

urls=(
 "https://suitesparse-collection-website.herokuapp.com/MM/Gleich/flickr.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian15.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/Andrianov/pattern1.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/Gupta/gupta3.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/gearbox.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/JGD_BIBD/bibd_22_8.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/TSOPF/TSOPF_FS_b300_c2.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/Meszaros/degme.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Ga19As19H42.tar.gz"
 "https://suitesparse-collection-website.herokuapp.com/MM/Andrianov/mip1.tar.gz"
)

for url in "${urls[@]}"; do
  filename="$(basename "$url")"
  archive="dataset/$filename"

  if [[ -f "$archive" ]]; then
    echo "$archive already exists"
  else
    wget -P dataset "$url"
  fi
done

for archive in dataset/*.tar.gz; do
  filename="$(basename "$archive")"
  dirname="${filename%.tar.gz}"
  outdir="dataset/$dirname"

  mkdir -p "$outdir"

  echo "extracting $archive"
  tar -xzf "$archive" -C "$outdir" --strip-components=1
done
