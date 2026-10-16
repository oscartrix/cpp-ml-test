#include "vec_classify/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace vec_classify {

Dataset load_csv_dataset(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo CSV: " + filepath);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Archivo CSV vacío: " + filepath);
    }

    // Contar columnas en el encabezado (f0, f1, ..., f{D-1}, label)
    std::size_t num_cols = 0;
    {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            ++num_cols;
        }
    }
    if (num_cols < 2) {
        throw std::runtime_error("Formato CSV inválido en: " + filepath);
    }
    const std::size_t dim = num_cols - 1;

    std::vector<float> flat_features;
    std::vector<int> labels;
    int max_label = -1;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        for (std::size_t d = 0; d < dim; ++d) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Fila incompleta en CSV: " + filepath);
            }
            if (token == "nan" || token == "NaN" || token == "NAN") {
                flat_features.push_back(std::numeric_limits<float>::quiet_NaN());
            } else {
                flat_features.push_back(std::strtof(token.c_str(), nullptr));
            }
        }
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error("Etiqueta faltante en CSV: " + filepath);
        }
        int lbl = std::atoi(token.c_str());
        labels.push_back(lbl);
        if (lbl > max_label) max_label = lbl;
    }

    Dataset ds;
    ds.X = Matrixf(labels.size(), dim);
    std::copy(flat_features.begin(), flat_features.end(), ds.X.data());
    ds.y = std::move(labels);
    ds.num_classes = static_cast<std::size_t>(max_label + 1);
    return ds;
}

double compute_macro_f1(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    std::size_t num_classes
) {
    if (y_true.size() != y_pred.size() || y_true.empty() || num_classes == 0) {
        return 0.0;
    }

    std::vector<std::size_t> tp(num_classes, 0);
    std::vector<std::size_t> fp(num_classes, 0);
    std::vector<std::size_t> fn(num_classes, 0);

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        int t = y_true[i];
        int p = y_pred[i];
        if (t < 0 || static_cast<std::size_t>(t) >= num_classes) continue;
        if (p == t) {
            ++tp[t];
        } else {
            ++fn[t];
            if (p >= 0 && static_cast<std::size_t>(p) < num_classes) {
                ++fp[p];
            }
        }
    }

    double f1_sum = 0.0;
    for (std::size_t c = 0; c < num_classes; ++c) {
        double denom = 2.0 * static_cast<double>(tp[c])
                     + static_cast<double>(fp[c])
                     + static_cast<double>(fn[c]);
        if (denom > 0.0) {
            f1_sum += (2.0 * static_cast<double>(tp[c])) / denom;
        }
    }
    return f1_sum / static_cast<double>(num_classes);
}

double compute_recall_at_k(
    const std::vector<std::vector<std::size_t>>& exact_neighbors,
    const std::vector<std::vector<std::size_t>>& approx_neighbors,
    std::size_t k
) {
    if (exact_neighbors.size() != approx_neighbors.size() || exact_neighbors.empty() || k == 0) {
        return 0.0;
    }

    double total_recall = 0.0;
    for (std::size_t q = 0; q < exact_neighbors.size(); ++q) {
        const auto& exact = exact_neighbors[q];
        const auto& approx = approx_neighbors[q];
        std::size_t limit = std::min(k, exact.size());
        if (limit == 0) continue;

        std::size_t hits = 0;
        for (std::size_t i = 0; i < std::min(k, approx.size()); ++i) {
            for (std::size_t j = 0; j < limit; ++j) {
                if (approx[i] == exact[j]) {
                    ++hits;
                    break;
                }
            }
        }
        total_recall += static_cast<double>(hits) / static_cast<double>(limit);
    }
    return total_recall / static_cast<double>(exact_neighbors.size());
}

} // namespace vec_classify
