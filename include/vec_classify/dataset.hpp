#pragma once

#include "vec_classify/tensor_view.hpp"
#include <string>
#include <vector>

namespace vec_classify {

struct Dataset {
    Matrixf X;
    std::vector<int> y;
    std::size_t num_classes{0};
};

/**
 * Carga un archivo CSV con formato f0,f1,...,f{D-1},label.
 * Nota: El archivo de entrenamiento puede contener lecturas de sensores corruptas
 * (NaN o valores centinela extremos).
 */
Dataset load_csv_dataset(const std::string& filepath);

/**
 * Calcula Macro-F1 sobre todas las clases presentes.
 */
double compute_macro_f1(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    std::size_t num_classes
);

/**
 * Calcula el Recall@K promedio del índice aproximado frente a los K vecinos exactos.
 */
double compute_recall_at_k(
    const std::vector<std::vector<std::size_t>>& exact_neighbors,
    const std::vector<std::vector<std::size_t>>& approx_neighbors,
    std::size_t k
);

} // namespace vec_classify
