#pragma once

#include "vec_classify/tensor_view.hpp"
#include <cstddef>
#include <vector>

namespace vec_classify {

/**
 * Interfaz base para métricas de distancia en R^D.
 *
 * NOTA PARA EL CANDIDATO:
 * Audita cuidadosamente esta interfaz base y las clases derivadas antes de implementar
 * nuevas métricas. Asegúrate de que el uso polimórfico a través de punteros inteligentes
 * sea seguro en memoria y eficiente en el bucle interno de búsqueda.
 */
class IDistanceMetric {
public:
    // [AUDIT]: Revisa el ciclo de vida polimórfico de esta clase base.
    ~IDistanceMetric() = default;

    /**
     * Calcula la distancia (o disimilitud) entre dos vectores `a` y `b` de dimensión `dim`.
     */
    virtual float compute(const float* a, const float* b, std::size_t dim) const = 0;

    /**
     * Calcula las distancias entre un vector `query` (1 x dim) y un bloque contiguo
     * de `n_candidates` vectores (n_candidates x dim), escribiendo el resultado en
     * `out_distances` (buffer pre-reservado de tamaño n_candidates).
     */
    virtual void compute_batch(
        const float* query,
        const float* candidates,
        std::size_t n_candidates,
        std::size_t dim,
        float* out_distances
    ) const {
        for (std::size_t i = 0; i < n_candidates; ++i) {
            out_distances[i] = compute(query, candidates + i * dim, dim);
        }
    }
};

/**
 * Distancia Euclídea al cuadrado (Squared L2):
 *   d(a, b) = \sum_{j=0}^{D-1} (a_j - b_j)^2
 */
class L2Metric : public IDistanceMetric {
public:
    float compute(const float* a, const float* b, std::size_t dim) const override;
};

/**
 * Disimilitud Coseno:
 *   d(a, b) = 1.0f - \frac{a \cdot b}{\max(\|a\|_2 \|b\|_2, \epsilon)}
 * donde \epsilon = 1e-8f para evitar división por cero.
 */
class CosineMetric : public IDistanceMetric {
public:
    float compute(const float* a, const float* b, std::size_t dim) const override;
};

/**
 * Distancia Euclídea Ponderada por Criterio de Fisher (Fisher-Weighted Squared L2):
 *   d_w(a, b) = \sum_{j=0}^{D-1} w_j (a_j - b_j)^2
 *
 * Los pesos `w_j` se aprenden de forma supervisada en `fit_weights(X, y)` usando la
 * razón entre la varianza inter-clase y la varianza intra-clase regularizada:
 *
 *   num_j = \sum_{c=0}^{C-1} \frac{N_c}{N_{valid}} (\mu_{c, j} - \mu_j)^2
 *   den_j = \frac{1}{N_{valid}} \sum_{c=0}^{C-1} \sum_{i \in c} (x_{i, j} - \mu_{c, j})^2 + \gamma
 *   s_j   = num_j / den_j
 *
 * Finalmente, el vector de pesos `w` se normaliza para que su suma sea igual a D:
 *   w_j = D * \frac{s_j}{\sum_{k=0}^{D-1} s_k + 1e-12}
 */
class FisherWeightedL2Metric : public IDistanceMetric {
public:
    explicit FisherWeightedL2Metric(float gamma = 1e-4f) : gamma_(gamma) {}

    /**
     * Aprende los pesos por dimensión ignorando cualquier fila corrupta (NaN o centinelas).
     */
    void fit_weights(const Tensor2DView& X, const std::vector<int>& y, std::size_t num_classes);

    float compute(const float* a, const float* b, std::size_t dim) const override;

    const std::vector<float>& weights() const noexcept { return weights_; }

private:
    float gamma_{1e-4f};
    std::vector<float> weights_;
};

} // namespace vec_classify
