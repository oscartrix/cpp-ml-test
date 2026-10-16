#include "vec_classify/classifier.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vec_classify {

AdaptiveKNNClassifier::AdaptiveKNNClassifier(ClassifierConfig config)
    : config_(config),
      metric_(std::make_shared<FisherWeightedL2Metric>()) {}

bool AdaptiveKNNClassifier::is_valid_row(const float* row, std::size_t dim) {
    // TODO (Etapa 2): Rechazar filas corruptas según la especificación del README.
    (void)row;
    (void)dim;
    return true;
}

void AdaptiveKNNClassifier::fit(
    const Tensor2DView& X,
    const std::vector<int>& y,
    std::size_t num_classes
) {
    if (X.rows() != y.size() || X.empty() || num_classes == 0) {
        throw std::invalid_argument("Datos de entrenamiento inválidos en AdaptiveKNNClassifier::fit");
    }
    num_classes_ = num_classes;
    const std::size_t d = X.cols();

    // 1. Filtrar filas corruptas usando `is_valid_row`
    std::vector<std::size_t> valid_rows;
    valid_rows.reserve(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (is_valid_row(X.row_ptr(i), d) && y[i] >= 0 && static_cast<std::size_t>(y[i]) < num_classes_) {
            valid_rows.push_back(i);
        }
    }

    clean_X_ = Matrixf(valid_rows.size(), d);
    clean_y_.resize(valid_rows.size());
    std::vector<std::size_t> class_counts(num_classes_, 0);

    for (std::size_t idx = 0; idx < valid_rows.size(); ++idx) {
        std::size_t orig = valid_rows[idx];
        std::copy(X.row_ptr(orig), X.row_ptr(orig) + d, clean_X_.row_ptr(idx));
        clean_y_[idx] = y[orig];
        ++class_counts[static_cast<std::size_t>(y[orig])];
    }

    // 2. Calcular priors empíricos \pi_c = N_c / N_clean
    class_priors_.assign(num_classes_, 1.0f / static_cast<float>(num_classes_));
    if (!clean_y_.empty()) {
        const float inv_n = 1.0f / static_cast<float>(clean_y_.size());
        for (std::size_t c = 0; c < num_classes_; ++c) {
            class_priors_[c] = std::max(static_cast<float>(class_counts[c]) * inv_n, 1e-6f);
        }
    }

    // 3. Ajustar pesos supervisados de Fisher
    metric_->fit_weights(clean_X_.view(), clean_y_, num_classes_);

    // 4. Construir el índice vectorial (IVF o Flat)
    if (config_.use_ivf) {
        index_ = std::make_unique<IVFIndex>(
            metric_,
            config_.n_lists,
            config_.n_probe
        );
    } else {
        index_ = std::make_unique<FlatIndex>(metric_);
    }
    index_->build(clean_X_.view());
}

int AdaptiveKNNClassifier::predict_one(const float* query, SearchResult& workspace) const {
    if (!index_ || num_classes_ == 0) {
        return 0;
    }

    index_->search(query, config_.k_neighbors, workspace);
    if (workspace.count == 0) {
        return 0;
    }

    // TODO (Etapa 4): Sustituir esta votación mayoritaria por la regla de decisión del
    // README, sin alocar memoria en el heap.
    float votes[32] = {0.0f};
    for (std::size_t i = 0; i < workspace.count; ++i) {
        std::size_t neighbor_idx = workspace.indices[i];
        int lbl = clean_y_[neighbor_idx];
        if (lbl >= 0 && static_cast<std::size_t>(lbl) < num_classes_ && lbl < 32) {
            votes[lbl] += 1.0f;
        }
    }

    int best_class = 0;
    float best_score = votes[0];
    for (std::size_t c = 1; c < std::min<std::size_t>(num_classes_, 32); ++c) {
        if (votes[c] > best_score) {
            best_score = votes[c];
            best_class = static_cast<int>(c);
        }
    }
    return best_class;
}

std::vector<int> AdaptiveKNNClassifier::predict_batch(const Tensor2DView& X_test) const {
    std::vector<int> preds(X_test.rows(), 0);
    SearchResult workspace;
    workspace.reserve(config_.k_neighbors, config_.n_lists);

    for (std::size_t i = 0; i < X_test.rows(); ++i) {
        preds[i] = predict_one(X_test.row_ptr(i), workspace);
    }
    return preds;
}

void AdaptiveKNNClassifier::set_nprobe(std::size_t n_probe) {
    config_.n_probe = n_probe;
    if (auto* ivf = dynamic_cast<IVFIndex*>(index_.get())) {
        ivf->set_nprobe(n_probe);
    }
}

} // namespace vec_classify
