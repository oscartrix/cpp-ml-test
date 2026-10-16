#include "vec_classify/alloc_tracker.hpp"
#include "vec_classify/classifier.hpp"
#include "vec_classify/dataset.hpp"
#include "vec_classify/distance.hpp"
#include "vec_classify/index.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

using namespace vec_classify;

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        ++g_failures;
    } else {
        std::cout << "[PASS] " << msg << "\n";
    }
}

void test_polymorphic_destructor() {
    bool has_virt = std::has_virtual_destructor<IDistanceMetric>::value;
    check(
        has_virt,
        "Etapa 1: IDistanceMetric debe tener destructor virtual para evitar fugas al destruir subclases"
    );
}

void test_flat_index_zero_allocations() {
    Matrixf X(128, 16);
    for (std::size_t i = 0; i < 128; ++i) {
        for (std::size_t j = 0; j < 16; ++j) {
            X(i, j) = static_cast<float>(i * 0.1 + j * 0.05);
        }
    }

    FlatIndex idx(std::make_shared<L2Metric>());
    idx.build(X.view());

    SearchResult out;
    out.reserve(8, 64);

    std::size_t allocs = 0;
    {
        ScopedAllocationTracker tracker;
        idx.search(X.row_ptr(10), 8, out);
        allocs = ScopedAllocationTracker::current_allocations();
    }

    check(out.count == 8 && out.indices[0] == 10, "Etapa 1: FlatIndex::search encuentra el vecino exacto en posicion 0");
    check(
        allocs == 0,
        "Etapa 1: FlatIndex::search ejecuta con CERO alocaciones de heap usando SearchResult pre-reservado"
    );
}

void test_cosine_and_fisher_metrics() {
    CosineMetric cos_m;
    float a[3] = {1.0f, 0.0f, 0.0f};
    float b[3] = {0.0f, 5.0f, 0.0f};
    float c[3] = {2.0f, 0.0f, 0.0f};

    float d_orth = cos_m.compute(a, b, 3);
    float d_same = cos_m.compute(a, c, 3);
    check(
        std::fabs(d_orth - 1.0f) < 1e-5f && std::fabs(d_same - 0.0f) < 1e-5f,
        "Etapa 2: CosineMetric calcula correctamente vectores ortogonales (1.0) y colineales (0.0)"
    );

    // Dataset sintético: dims 0..1 separan perfectamente las clases 0 y 1; dims 2..3 son ruido masivo
    std::mt19937 rng(123);
    std::normal_distribution<float> small_noise(0.0f, 0.2f);
    std::normal_distribution<float> huge_noise(0.0f, 500.0f);

    Matrixf X(200, 4);
    std::vector<int> y(200);
    for (std::size_t i = 0; i < 200; ++i) {
        int cls = (i < 100) ? 0 : 1;
        y[i] = cls;
        float center = (cls == 0) ? -5.0f : 5.0f;
        X(i, 0) = center + small_noise(rng);
        X(i, 1) = center * 0.8f + small_noise(rng);
        X(i, 2) = huge_noise(rng);
        X(i, 3) = huge_noise(rng);
    }

    FisherWeightedL2Metric fisher;
    fisher.fit_weights(X.view(), y, 2);
    const auto& w = fisher.weights();

    bool weights_ok = (w.size() == 4)
                   && (w[0] > 1.0f && w[1] > 1.0f)
                   && (w[2] < 0.01f && w[3] < 0.01f);
    check(
        weights_ok,
        "Etapa 2: FisherWeightedL2Metric asigna peso alto a dimensiones informativas y ~0 a dimensiones de ruido extremo"
    );
}

void test_sensor_corruption_filter() {
    float good[4] = {1.2f, -3.4f, 120.0f, 0.0f};
    float bad_nan[4] = {1.2f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
    float bad_sentinel[4] = {0.5f, -9999.0f, 1.0f, 2.0f};

    bool ok = AdaptiveKNNClassifier::is_valid_row(good, 4)
           && !AdaptiveKNNClassifier::is_valid_row(bad_nan, 4)
           && !AdaptiveKNNClassifier::is_valid_row(bad_sentinel, 4);
    check(
        ok,
        "Etapa 2: AdaptiveKNNClassifier::is_valid_row detecta y rechaza filas con NaN y centinelas (-9999.0)"
    );
}

void test_ivf_index_recall_and_zero_allocs() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const std::size_t n_clusters = 8;
    const std::size_t pts_per_cluster = 50;
    const std::size_t n = n_clusters * pts_per_cluster;
    const std::size_t d = 8;

    Matrixf X(n, d);
    for (std::size_t c = 0; c < n_clusters; ++c) {
        for (std::size_t p = 0; p < pts_per_cluster; ++p) {
            std::size_t row = c * pts_per_cluster + p;
            for (std::size_t j = 0; j < d; ++j) {
                float center = (j == (c % d)) ? 15.0f : -2.0f * static_cast<float>(c);
                X(row, j) = center + dist(rng);
            }
        }
    }

    auto metric = std::make_shared<L2Metric>();
    FlatIndex flat(metric);
    flat.build(X.view());

    IVFIndex ivf(metric, n_clusters, 3, 20, 42);
    ivf.build(X.view());

    SearchResult flat_out, ivf_out;
    flat_out.reserve(5, n_clusters);
    ivf_out.reserve(5, n_clusters);

    std::size_t allocs = 0;
    std::vector<std::vector<std::size_t>> exact_res(40), approx_res(40);
    for (std::size_t q = 0; q < 40; ++q) {
        const float* q_ptr = X.row_ptr(q * 10);
        flat.search(q_ptr, 5, flat_out);
        {
            ScopedAllocationTracker tracker;
            ivf.search(q_ptr, 5, ivf_out);
            allocs += ScopedAllocationTracker::current_allocations();
        }
        exact_res[q].assign(flat_out.indices.begin(), flat_out.indices.begin() + static_cast<std::ptrdiff_t>(flat_out.count));
        approx_res[q].assign(ivf_out.indices.begin(), ivf_out.indices.begin() + static_cast<std::ptrdiff_t>(ivf_out.count));
    }

    double recall = compute_recall_at_k(exact_res, approx_res, 5);
    check(
        ivf_out.count == 5 && recall >= 0.85,
        "Etapa 3: IVFIndex con k-Means++ alcanza Recall@5 >= 0.85 en clusters bien definidos"
    );
    // Exige resultados reales: un `search()` vacío no debe pasar este test
    check(
        allocs == 0 && ivf_out.count == 5,
        "Etapa 3: IVFIndex::search ejecuta con CERO alocaciones de heap por consulta"
    );

    // Métrica instrumentada para verificar que IVF no recorre todo el dataset.
    // Hereda de IDistanceMetric (no de L2Metric) para que `compute_batch` caiga en `compute`.
    struct CountingL2Metric : IDistanceMetric {
        mutable std::size_t calls{0};
        float compute(const float* a, const float* b, std::size_t dim) const override {
            ++calls;
            float sum = 0.0f;
            for (std::size_t j = 0; j < dim; ++j) {
                float diff = a[j] - b[j];
                sum += diff * diff;
            }
            return sum;
        }
    };

    auto counting = std::make_shared<CountingL2Metric>();
    IVFIndex ivf_counted(counting, n_clusters, 1, 20, 42);
    ivf_counted.build(X.view());
    counting->calls = 0;
    for (std::size_t q = 0; q < 40; ++q) {
        ivf_counted.search(X.row_ptr(q * 10), 5, ivf_out);
    }
    const double calls_per_query = static_cast<double>(counting->calls) / 40.0;
    check(
        calls_per_query > 0.0 && calls_per_query < static_cast<double>(n) / 2.0,
        "Etapa 3: IVFIndex::search con nprobe=1 evalúa menos de la mitad del dataset por consulta"
    );
}

void test_weighted_vote_rule() {
    // 1D, Flat: la consulta en 0.0 tiene 1 vecino de clase 1 a distancia 0 y 2 vecinos de
    // clase 0 a distancia ~9. La mayoría simple elige 0; el kernel Gaussiano debe elegir 1.
    {
        Matrixf X(3, 1);
        X(0, 0) = 0.0f;
        X(1, 0) = 3.0f;
        X(2, 0) = 3.1f;
        std::vector<int> y = {1, 0, 0};

        ClassifierConfig cfg;
        cfg.use_ivf = false;
        cfg.k_neighbors = 3;
        cfg.temperature = 1.0f;
        cfg.class_prior_alpha = 0.0f;
        AdaptiveKNNClassifier clf(cfg);
        clf.fit(X.view(), y, 2);

        SearchResult ws;
        ws.reserve(cfg.k_neighbors, cfg.n_lists);
        float q = 0.0f;
        check(
            clf.predict_one(&q, ws) == 1,
            "Etapa 4: predict_one pondera los votos con el kernel Gaussiano de distancia"
        );
    }

    // Dos vecinos equidistantes de clases distintas: el prior inverso debe favorecer a la
    // clase minoritaria (1 de 5 muestras de entrenamiento).
    {
        Matrixf X(5, 1);
        X(0, 0) = 1.0f;
        X(1, 0) = 50.0f;
        X(2, 0) = 51.0f;
        X(3, 0) = 52.0f;
        X(4, 0) = -1.0f;
        std::vector<int> y = {0, 0, 0, 0, 1};

        ClassifierConfig cfg;
        cfg.use_ivf = false;
        cfg.k_neighbors = 2;
        cfg.temperature = 1.0f;
        cfg.class_prior_alpha = 1.0f;
        AdaptiveKNNClassifier clf(cfg);
        clf.fit(X.view(), y, 2);

        SearchResult ws;
        ws.reserve(cfg.k_neighbors, cfg.n_lists);
        float q = 0.0f;
        std::size_t allocs = 0;
        int pred = -1;
        {
            ScopedAllocationTracker tracker;
            pred = clf.predict_one(&q, ws);
            allocs = ScopedAllocationTracker::current_allocations();
        }
        check(
            pred == 1,
            "Etapa 4: predict_one compensa el desbalance con el prior inverso (pi_c)^-alpha"
        );
        check(
            allocs == 0,
            "Etapa 4: AdaptiveKNNClassifier::predict_one ejecuta con CERO alocaciones de heap"
        );
    }
}

} // namespace

int main() {
    std::cout << "=== Ejecutando Suite de Auditoria y Pruebas Unitarias ===\n";
    test_polymorphic_destructor();
    test_flat_index_zero_allocations();
    test_cosine_and_fisher_metrics();
    test_sensor_corruption_filter();
    test_ivf_index_recall_and_zero_allocs();
    test_weighted_vote_rule();

    std::cout << "---------------------------------------------------------\n";
    if (g_failures == 0) {
        std::cout << "TODOS LOS TESTS UNITARIOS PASARON EXITOSAMENTE.\n";
        return 0;
    } else {
        std::cout << "FALLARON " << g_failures << " TESTS. Revisa los TODOs y las notas [AUDIT].\n";
        return 1;
    }
}
