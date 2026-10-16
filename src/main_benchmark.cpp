#include "vec_classify/alloc_tracker.hpp"
#include "vec_classify/classifier.hpp"
#include "vec_classify/dataset.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace vec_classify;

    std::string train_path = "data/train.csv";
    std::string test_path = "data/test.csv";
    if (argc >= 3) {
        train_path = argv[1];
        test_path = argv[2];
    }

    std::cout << "=== VecClassify C++ & ML Benchmark ===\n";
    Dataset train_ds = load_csv_dataset(train_path);
    Dataset test_ds = load_csv_dataset(test_path);

    std::cout << "Train cargado: " << train_ds.X.rows() << " muestras x "
              << train_ds.X.cols() << " dims (" << train_ds.num_classes << " clases)\n";
    std::cout << "Test cargado : " << test_ds.X.rows() << " muestras x "
              << test_ds.X.cols() << " dims\n\n";

    // 1. Entrenar clasificador base (FlatIndex exacto) para referencia de Recall@10 y tiempo
    ClassifierConfig flat_cfg;
    flat_cfg.use_ivf = false;
    flat_cfg.k_neighbors = 11;
    AdaptiveKNNClassifier flat_clf(flat_cfg);
    flat_clf.fit(train_ds.X.view(), train_ds.y, train_ds.num_classes);

    std::cout << "Muestras válidas tras filtrado de sensores corruptos: "
              << flat_clf.clean_train_size() << " / " << train_ds.X.rows() << "\n";

    if (flat_clf.metric() && !flat_clf.metric()->weights().empty()) {
        const auto& w = flat_clf.metric()->weights();
        std::cout << "Pesos Fisher (primeras 4 dims de señal vs últimas 4 dims de ruido):\n  Señal [0..3]: ";
        for (std::size_t j = 0; j < 4 && j < w.size(); ++j) {
            std::cout << std::fixed << std::setprecision(3) << w[j] << " ";
        }
        std::cout << "\n  Ruido [28..31]: ";
        for (std::size_t j = w.size() - 4; j < w.size(); ++j) {
            std::cout << std::fixed << std::setprecision(5) << w[j] << " ";
        }
        std::cout << "\n\n";
    }

    // Obtener vecinos exactos (K=10) y latencia de FlatIndex
    const std::size_t k_eval = 10;
    std::vector<std::vector<std::size_t>> exact_neighbors(test_ds.X.rows());
    std::vector<int> flat_preds(test_ds.X.rows(), 0);
    SearchResult workspace;
    workspace.reserve(flat_cfg.k_neighbors, flat_cfg.n_lists);

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < test_ds.X.rows(); ++i) {
        flat_preds[i] = flat_clf.predict_one(test_ds.X.row_ptr(i), workspace);
        exact_neighbors[i].assign(
            workspace.indices.begin(),
            workspace.indices.begin() + static_cast<std::ptrdiff_t>(std::min(k_eval, workspace.count))
        );
    }
    auto t1 = std::chrono::steady_clock::now();
    double flat_us_per_query = std::chrono::duration<double, std::micro>(t1 - t0).count()
                             / static_cast<double>(test_ds.X.rows());
    double flat_f1 = compute_macro_f1(test_ds.y, flat_preds, train_ds.num_classes);

    std::cout << "Referencia Exacta (FlatIndex): Macro-F1 = " << std::fixed << std::setprecision(4)
              << flat_f1 << " | Latencia = " << std::setprecision(2) << flat_us_per_query << " us/query\n\n";

    // 2. Entrenar clasificador con IVFIndex y evaluar para distintos valores de nprobe
    ClassifierConfig ivf_cfg;
    ivf_cfg.use_ivf = true;
    ivf_cfg.n_lists = 64;
    ivf_cfg.k_neighbors = 11;

    AdaptiveKNNClassifier ivf_clf(ivf_cfg);
    ivf_clf.fit(train_ds.X.view(), train_ds.y, train_ds.num_classes);

    std::cout << "| nprobe | Macro-F1 | Recall@10 | Latencia (us/q) | Speedup vs Flat | Heap Allocs/q |\n";
    std::cout << "|-------:|---------:|----------:|----------------:|----------------:|--------------:|\n";

    // Umbrales de aceptación evaluados con nprobe = 8 (ver README)
    const std::size_t accept_nprobe = 8;
    const double min_f1 = 0.87;
    const double min_recall = 0.80;
    const double min_speedup = 3.0;
    bool accept_ok = false;

    const std::vector<std::size_t> probes = {1, 2, 4, 8, 16, 32, 64};
    for (std::size_t np : probes) {
        ivf_clf.set_nprobe(np);
        std::vector<std::vector<std::size_t>> approx_neighbors(test_ds.X.rows());
        std::vector<int> preds(test_ds.X.rows(), 0);
        workspace.reserve(ivf_cfg.k_neighbors, ivf_cfg.n_lists);

        std::size_t total_allocs = 0;
        auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < test_ds.X.rows(); ++i) {
            {
                ScopedAllocationTracker tracker;
                preds[i] = ivf_clf.predict_one(test_ds.X.row_ptr(i), workspace);
                total_allocs += ScopedAllocationTracker::current_allocations();
            }
            approx_neighbors[i].assign(
                workspace.indices.begin(),
                workspace.indices.begin() + static_cast<std::ptrdiff_t>(std::min(k_eval, workspace.count))
            );
        }
        auto end = std::chrono::steady_clock::now();

        double us_per_q = std::chrono::duration<double, std::micro>(end - start).count()
                        / static_cast<double>(test_ds.X.rows());
        double f1 = compute_macro_f1(test_ds.y, preds, train_ds.num_classes);
        double recall = compute_recall_at_k(exact_neighbors, approx_neighbors, k_eval);
        double speedup = (us_per_q > 0.0) ? (flat_us_per_query / us_per_q) : 0.0;
        double allocs_per_q = static_cast<double>(total_allocs) / static_cast<double>(test_ds.X.rows());

        std::cout << "| " << std::setw(6) << np
                  << " | " << std::setw(8) << std::fixed << std::setprecision(4) << f1
                  << " | " << std::setw(9) << std::fixed << std::setprecision(4) << recall
                  << " | " << std::setw(15) << std::fixed << std::setprecision(2) << us_per_q
                  << " | " << std::setw(14) << std::fixed << std::setprecision(2) << speedup << "x"
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1) << allocs_per_q
                  << " |\n";

        if (np == accept_nprobe) {
            accept_ok = f1 >= min_f1 && recall >= min_recall && speedup >= min_speedup && total_allocs == 0;
        }
    }

    std::cout << std::setprecision(2)
              << "\nCriterios de aceptación (nprobe = " << accept_nprobe << "): Macro-F1 >= " << min_f1
              << ", Recall@10 >= " << min_recall << ", Speedup >= " << min_speedup
              << "x, Heap Allocs/q = 0 -> " << (accept_ok ? "CUMPLE" : "NO CUMPLE") << "\n";
    return accept_ok ? 0 : 2;
}
