#include "vec_classify/index.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace vec_classify {

FlatIndex::FlatIndex(std::shared_ptr<const IDistanceMetric> metric)
    : metric_(std::move(metric)) {
    if (!metric_) {
        throw std::invalid_argument("FlatIndex requiere una métrica no nula.");
    }
}

void FlatIndex::build(const Tensor2DView& data) {
    data_ = Matrixf(data.rows(), data.cols());
    if (!data.empty()) {
        std::copy(
            data.data(),
            data.data() + data.rows() * data.cols(),
            data_.data()
        );
    }
}

void FlatIndex::search(
    const float* query,
    std::size_t k,
    SearchResult& out
) const {
    const std::size_t n = data_.rows();
    const std::size_t d = data_.cols();
    if (n == 0 || k == 0) {
        out.count = 0;
        return;
    }

    // [AUDIT - Etapa 1]: Revisa esta función frente a la restricción de 0 alocaciones.
    std::vector<std::pair<float, std::size_t>> all_candidates;
    all_candidates.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> row_copy(data_.row_ptr(i), data_.row_ptr(i) + d);
        float dist = metric_->compute(query, row_copy.data(), d);
        all_candidates.push_back({dist, i});
    }

    const std::size_t actual_k = std::min(k, n);
    std::partial_sort(
        all_candidates.begin(),
        all_candidates.begin() + static_cast<std::ptrdiff_t>(actual_k),
        all_candidates.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; }
    );

    if (out.indices.size() < actual_k) out.indices.resize(actual_k);
    if (out.distances.size() < actual_k) out.distances.resize(actual_k);

    for (std::size_t i = 0; i < actual_k; ++i) {
        out.distances[i] = all_candidates[i].first;
        out.indices[i] = all_candidates[i].second;
    }
    out.count = actual_k;
}

IVFIndex::IVFIndex(
    std::shared_ptr<const IDistanceMetric> metric,
    std::size_t n_lists,
    std::size_t n_probe,
    std::size_t max_kmeans_iters,
    unsigned int random_seed
)
    : metric_(std::move(metric)),
      n_lists_(n_lists),
      n_probe_(n_probe),
      max_kmeans_iters_(max_kmeans_iters),
      seed_(random_seed) {
    if (!metric_) {
        throw std::invalid_argument("IVFIndex requiere una métrica no nula.");
    }
}

void IVFIndex::build(const Tensor2DView& data) {
    total_vectors_ = data.rows();
    dim_ = data.cols();
    if (total_vectors_ == 0 || dim_ == 0) {
        return;
    }

    // TODO (Etapa 3): Implementar el entrenamiento descrito en el README. Debe ser
    // determinista dado `seed_` y robusto ante casos límite (p. ej. clusters vacíos).
    (void)max_kmeans_iters_;
    (void)seed_;
}

void IVFIndex::search(
    const float* query,
    std::size_t k,
    SearchResult& out
) const {
    // TODO (Etapa 3): Implementar la búsqueda descrita en el README y en `VectorIndex::search`.
    (void)query;
    (void)k;
    out.count = 0;
}

} // namespace vec_classify
