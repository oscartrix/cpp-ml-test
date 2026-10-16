#include "vec_classify/distance.hpp"
#include <cmath>
#include <numeric>

namespace vec_classify {

float L2Metric::compute(const float* a, const float* b, std::size_t dim) const {
    float sum = 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        float diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

float CosineMetric::compute(const float* a, const float* b, std::size_t dim) const {
    // TODO (Etapa 2): Implementar la disimilitud coseno definida en distance.hpp.
    (void)a;
    (void)b;
    (void)dim;
    return 0.0f;
}

void FisherWeightedL2Metric::fit_weights(
    const Tensor2DView& X,
    const std::vector<int>& y,
    std::size_t num_classes
) {
    // TODO (Etapa 2): Implementar el criterio de Fisher definido en distance.hpp.
    // Baseline inicial (sin ponderar: todos los pesos = 1.0f):
    (void)y;
    (void)num_classes;
    weights_.assign(X.cols(), 1.0f);
}

float FisherWeightedL2Metric::compute(const float* a, const float* b, std::size_t dim) const {
    // TODO (Etapa 2): Revisar contra la definición de distance.hpp.
    float sum = 0.0f;
    const bool has_weights = (weights_.size() == dim);
    for (std::size_t j = 0; j < dim; ++j) {
        float diff = a[j] - b[j];
        float w = has_weights ? weights_[j] : 1.0f;
        sum += w * diff * diff;
    }
    return sum;
}

} // namespace vec_classify
