#pragma once

#include "vec_classify/distance.hpp"
#include "vec_classify/index.hpp"
#include "vec_classify/tensor_view.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace vec_classify {

struct ClassifierConfig {
    std::size_t k_neighbors{11};
    std::size_t n_lists{64};
    std::size_t n_probe{8};
    float temperature{5.0f};      // \tau en el kernel Gaussiano de votación
    float class_prior_alpha{0.6f}; // Exponente \alpha para compensar desbalance (\pi_c^{-\alpha})
    bool use_ivf{true};
};

/**
 * Clasificador k-NN Adaptativo acelerado por índice vectorial (Flat o IVF).
 *
 * Flujo de entrenamiento (`fit`):
 * 1. Sanea el dataset de entrenamiento descartando muestras que contengan `NaN`, `Inf`
 *    o valores centinela de sensor fuera del rango físico razonable en las dimensiones
 *    de señal (por ejemplo, valores exactos como `-9999.0f`).
 * 2. Calcula los priors empíricos de cada clase `\pi_c = N_c / N_{clean}`.
 * 3. Ajusta los pesos supervisados de `FisherWeightedL2Metric` sobre los datos limpios.
 * 4. Construye el índice vectorial (`IVFIndex` o `FlatIndex`) usando la métrica aprendida.
 *
 * Regla de decisión en inferencia (`predict_one`):
 * Dado un vector de consulta `q` y sus `k` vecinos más cercanos `(idx_i, dist_i)` donde
 * `dist_i` es la distancia Fisher-Weighted Squared L2 `d_w(q, x_i)`:
 *
 *   Score(c | q) = (\pi_c)^{-\alpha} * \sum_{i \in N_k(q), y_i = c} \exp( - dist_i / (2 * \tau^2) )
 *
 * Retorna la clase `c` que maximiza `Score(c | q)`.
 */
class AdaptiveKNNClassifier {
public:
    explicit AdaptiveKNNClassifier(ClassifierConfig config = {});

    /**
     * Entrena la métrica ponderada, filtra filas corruptas y construye el índice.
     */
    void fit(const Tensor2DView& X, const std::vector<int>& y, std::size_t num_classes);

    /**
     * Predice la etiqueta de clase para un único vector `query` usando un `SearchResult`
     * pre-reservado (0 alocaciones de heap).
     */
    int predict_one(const float* query, SearchResult& workspace) const;

    /**
     * Predice las etiquetas para un lote de consultas `X_test`.
     */
    std::vector<int> predict_batch(const Tensor2DView& X_test) const;

    void set_nprobe(std::size_t n_probe);

    std::size_t clean_train_size() const noexcept { return clean_y_.size(); }
    const FisherWeightedL2Metric* metric() const noexcept { return metric_.get(); }
    const VectorIndex* index() const noexcept { return index_.get(); }

    /**
     * Determina si una fila de dimensión `dim` es válida (no contiene NaN/Inf ni
     * el valor centinela de desconexión de sensor `-9999.0`).
     */
    static bool is_valid_row(const float* row, std::size_t dim);

private:
    ClassifierConfig config_;
    std::size_t num_classes_{0};
    Matrixf clean_X_;
    std::vector<int> clean_y_;
    std::vector<float> class_priors_;
    std::shared_ptr<FisherWeightedL2Metric> metric_;
    std::unique_ptr<VectorIndex> index_;
};

} // namespace vec_classify
