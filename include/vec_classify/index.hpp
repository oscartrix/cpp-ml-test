#pragma once

#include "vec_classify/distance.hpp"
#include "vec_classify/tensor_view.hpp"
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace vec_classify {

/**
 * Buffer de salida y espacio de trabajo pre-reservado para consultas `search()`.
 *
 * Para cumplir la restricción de CERO alocaciones dinámicas (0 heap allocations)
 * en el camino crítico (`search`), las implementaciones de `VectorIndex` deben
 * operar exclusivamente sobre la memoria ya reservada en `SearchResult` (o memoria
 * en el stack de tamaño fijo).
 */
struct SearchResult {
    std::size_t count{0};
    std::vector<std::size_t> indices;
    std::vector<float> distances;

    // Espacio temporal opcional pre-reservado para ordenar centroides en IVF sin alocar en heap
    std::vector<std::pair<float, std::size_t>> probe_scratch;

    void reserve(std::size_t max_k, std::size_t max_lists = 256) {
        indices.resize(max_k);
        distances.resize(max_k);
        probe_scratch.resize(max_lists);
        count = 0;
    }
};

/**
 * Interfaz abstracta para índices de búsqueda de vecinos más cercanos (ANN / Exacto).
 */
class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    /**
     * Construye el índice a partir de un conjunto de vectores válidos (N x D).
     */
    virtual void build(const Tensor2DView& data) = 0;

    /**
     * Busca los `k` vecinos más cercanos al vector `query` (dimensión D).
     * Los resultados deben quedar ordenados de menor a mayor distancia en `out.indices[0..count-1]`
     * y `out.distances[0..count-1]`, donde `count = min(k, num_indexed_vectors)`.
     *
     * RESTRICCIÓN DE RENDIMIENTO:
     * Si `out.reserve(k, n_lists)` fue llamado previamente, `search()` NO debe realizar
     * ninguna alocación en el heap (`new` / `malloc` = 0).
     */
    virtual void search(
        const float* query,
        std::size_t k,
        SearchResult& out
    ) const = 0;

    virtual std::size_t size() const noexcept = 0;
    virtual std::size_t dim() const noexcept = 0;
};

/**
 * Índice de fuerza bruta (búsqueda exacta sobre todos los vectores).
 *
 * NOTA PARA EL CANDIDATO:
 * La implementación inicial en `src/index.cpp` es funcional pero viola la restricción
 * de cero alocaciones en `search()` y realiza copias innecesarias. Debes refactorizarla.
 */
class FlatIndex : public VectorIndex {
public:
    explicit FlatIndex(std::shared_ptr<const IDistanceMetric> metric);

    void build(const Tensor2DView& data) override;

    void search(
        const float* query,
        std::size_t k,
        SearchResult& out
    ) const override;

    std::size_t size() const noexcept override { return data_.rows(); }
    std::size_t dim() const noexcept override { return data_.cols(); }

private:
    std::shared_ptr<const IDistanceMetric> metric_;
    Matrixf data_;
};

/**
 * Índice de Archivo Invertido (Inverted File Index - IVF).
 *
 * Entrena `n_lists` centroides mediante k-Means++ usando la métrica configurada,
 * particiona los vectores en `n_lists` listas invertidas (celdas de Voronoi), y
 * durante `search()` evalúa únicamente los vectores pertenecientes a las `n_probe`
 * celdas cuyos centroides estén más cercanos al `query`.
 */
class IVFIndex : public VectorIndex {
public:
    IVFIndex(
        std::shared_ptr<const IDistanceMetric> metric,
        std::size_t n_lists = 64,
        std::size_t n_probe = 8,
        std::size_t max_kmeans_iters = 20,
        unsigned int random_seed = 42
    );

    void build(const Tensor2DView& data) override;

    void search(
        const float* query,
        std::size_t k,
        SearchResult& out
    ) const override;

    void set_nprobe(std::size_t n_probe) noexcept { n_probe_ = n_probe; }
    std::size_t nprobe() const noexcept { return n_probe_; }
    std::size_t nlists() const noexcept { return n_lists_; }

    std::size_t size() const noexcept override { return total_vectors_; }
    std::size_t dim() const noexcept override { return dim_; }

private:
    std::shared_ptr<const IDistanceMetric> metric_;
    std::size_t n_lists_{64};
    std::size_t n_probe_{8};
    std::size_t max_kmeans_iters_{20};
    unsigned int seed_{42};

    std::size_t total_vectors_{0};
    std::size_t dim_{0};

    // Centroides de tamaño (n_lists_ x dim_)
    Matrixf centroids_;

    // Listas invertidas: para cada cluster c in [0..n_lists_-1],
    // puedes almacenar los vectores de esa celda de forma contigua en `list_vectors_[c]`
    // y sus IDs originales [0..N-1] en `list_ids_[c]`.
    std::vector<Matrixf> list_vectors_;
    std::vector<std::vector<std::size_t>> list_ids_;
};

} // namespace vec_classify
