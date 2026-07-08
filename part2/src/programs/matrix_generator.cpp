#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

enum matrix_types { none, real_general, pattern_general };

struct nnz_entry {
  int row;
  int col;
  float val;
};

int comp_nnz_entry(const void *left, const void *right) {
  if (((nnz_entry *)left)->col < ((nnz_entry *)right)->col) {
    return -1;
  }

  if (((nnz_entry *)left)->col > ((nnz_entry *)right)->col) {
    return 1;
  }

  return 0;
}

matrix_types parse_matrix_type(const char *arg) {
  if (std::strcmp(arg, "real_general") == 0) {
    return real_general;
  }

  if (std::strcmp(arg, "pattern_general") == 0) {
    return pattern_general;
  }

  return none;
}

void generate(nnz_entry **entries_out, const uint64_t rows, const uint64_t cols,
              uint64_t &nnz, const float max_over_avg) {
  uint64_t active_rows = nnz < rows ? nnz : rows;
  // at most nnz / rows, at minimum 1
  double expected_avg =
      static_cast<double>(nnz) / static_cast<double>(active_rows);

  // going over is okay
  uint64_t avg = static_cast<uint64_t>(std::ceil(expected_avg));
  uint64_t max = static_cast<uint64_t>(max_over_avg * avg);

  // unbalanced row cannot have less than avg elements
  if (max < avg) {
    max = avg;
  }

  // it also cannot have more elements than there are columns
  if (max > cols) {
    max = cols;
  }

  std::unordered_set<uint64_t> active_row_ids;
  for (uint64_t i = 0; i < active_rows; i++) {
    uint64_t random_row = rand() % rows;

    while (!active_row_ids.insert(random_row).second) {
      random_row = rand() % rows;
    }
  }

  std::vector<uint64_t> active_row_ids_vec(active_row_ids.begin(),
                                           active_row_ids.end());

  uint64_t *counts =
      reinterpret_cast<uint64_t *>(malloc(sizeof(uint64_t) * rows));
  std::memset(counts, 0, sizeof(uint64_t) * rows);
  for (const auto &active_row_id : active_row_ids_vec) {
    counts[active_row_id] = avg;
  }

  // now that we have our counts, we can start syphoning values off other rows
  // uniformly to make our unbalanced row
  auto unbalanced_row = active_row_ids_vec[rand() % active_row_ids_vec.size()];
  for (uint64_t i = 0; i < max - avg; i++) {
    auto rand_row = active_row_ids_vec[rand() % active_row_ids_vec.size()];
    // make sure the row does have an element
    while ((counts[rand_row] <= 0) || rand_row == unbalanced_row) {
      rand_row = active_row_ids_vec[rand() % active_row_ids_vec.size()];
    }
    // take it
    counts[rand_row]--;
    // give it to the unbalanced one
    counts[unbalanced_row]++;
  }

  // now we have a list of rows that have either 0 or 1 as num of elements in
  // case the matrix is extremely sparse or an avg on every row
  uint64_t sum_of_counts = std::reduce(counts, counts + rows, uint64_t(0));
  std::cout << "sum of counts: " << sum_of_counts << std::endl;

  // in case nnz varies
  nnz = sum_of_counts;

  uint64_t writing_idx = 0;
  nnz_entry *entries =
      reinterpret_cast<nnz_entry *>(malloc(sizeof(nnz_entry) * nnz));
  std::memset(entries, 0, sizeof(nnz_entry) * nnz);

  for (uint64_t i = 0; i < active_row_ids_vec.size(); i++) {
    std::unordered_set<uint64_t> extracted_cols;
    for (uint64_t j = 0; j < counts[active_row_ids_vec[i]]; j++) {
      uint64_t row = active_row_ids_vec[i];
      uint64_t col = rand() % cols;
      while (!extracted_cols.insert(col).second) {
        col = rand() % cols;
      }
      entries[writing_idx].row = row + 1;
      entries[writing_idx].col = col + 1;
      entries[writing_idx].val =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      writing_idx++;
    }
  }

  // the files from SuiteSparse seem to be sorted by column coordinate
  qsort(entries, nnz, sizeof(nnz_entry), comp_nnz_entry);

  free(counts);
  *entries_out = entries;
}

int main(int argc, const char *argv[]) {
  if (argc != 8) {
    std::cout << "Usage: " << argv[0]
              << "<rows> <cols> <nnz> <pattern_general|real_general> "
                 "<max_over_avg> <seed> <filename>"
              << std::endl;
    return 1;
  }

  uint64_t rows = static_cast<uint64_t>(std::stoi(argv[1]));
  uint64_t cols = static_cast<uint64_t>(std::stoi(argv[2]));
  uint64_t nnz = static_cast<uint64_t>(std::stoi(argv[3]));
  auto type = parse_matrix_type(argv[4]);
  auto max_over_avg = std::stof(argv[5]);
  auto seed = std::stoi(argv[6]);
  auto filename = argv[7];

  std::cout << "nnz: " << nnz << " max_over_avg: " << max_over_avg
            << " seed: " << seed << " filename: " << filename << std::endl;

  srand(seed);

  assert(nnz < (rows * cols));

  std::cout << "nnz will fit, predicted sparsity: "
            << (1.0 -
                (static_cast<double>(nnz) / static_cast<double>(rows * cols)))
            << std::endl;

  std::cout << "rows: " << rows << " cols: " << cols << " nnz: " << nnz
            << std::endl;

  nnz_entry *entries = nullptr;
  // this will further adjust nnz
  generate(&entries, rows, cols, nnz, max_over_avg);

  std::ofstream file(filename);
  assert(file.good());

  switch (type) {
  case real_general:
    file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    break;
  case pattern_general:
    file << "%%MatrixMarket matrix coordinate pattern general" << std::endl;
    break;
  default:
    assert(false);
    break;
  }

  file << "% nnz: " << nnz << " max_over_avg: " << max_over_avg
       << " seed: " << seed << " filename: " << filename << std::endl;

  file << rows << " " << cols << " " << nnz << std::endl;

  switch (type) {
  case real_general:
    for (uint64_t i = 0; i < nnz; i++) {
      file << entries[i].row << " " << entries[i].col << " " << entries[i].val
           << std::endl;
    }
    break;
  case pattern_general:
    for (uint64_t i = 0; i < nnz; i++) {
      file << entries[i].row << " " << entries[i].col << std::endl;
    }
    break;
  default:
    assert(false);
    break;
  }

  free(entries);
}
