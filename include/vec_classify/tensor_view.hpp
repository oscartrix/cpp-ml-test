#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace vec_classify {

/**
 * Vista ligera no propietaria sobre una matriz 2D contigua en memoria (row-major).
 * Evita copias innecesarias y reemplaza el antipatrón `std::vector<std::vector<float>>`.
 */
class Tensor2DView {
public:
    Tensor2DView() = default;
    Tensor2DView(const float* data, std::size_t rows, std::size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    const float* data() const noexcept { return data_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    bool empty() const noexcept { return rows_ == 0 || cols_ == 0 || data_ == nullptr; }

    const float* row_ptr(std::size_t r) const noexcept {
        assert(r < rows_);
        return data_ + r * cols_;
    }

    float operator()(std::size_t r, std::size_t c) const noexcept {
        assert(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }

private:
    const float* data_{nullptr};
    std::size_t rows_{0};
    std::size_t cols_{0};
};

/**
 * Contenedor propietario de una matriz 2D contigua en memoria (row-major).
 */
class Matrixf {
public:
    Matrixf() = default;
    Matrixf(std::size_t rows, std::size_t cols, float init_val = 0.0f)
        : rows_(rows), cols_(cols), storage_(rows * cols, init_val) {}

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    bool empty() const noexcept { return storage_.empty(); }

    float* data() noexcept { return storage_.data(); }
    const float* data() const noexcept { return storage_.data(); }

    float* row_ptr(std::size_t r) noexcept {
        assert(r < rows_);
        return storage_.data() + r * cols_;
    }

    const float* row_ptr(std::size_t r) const noexcept {
        assert(r < rows_);
        return storage_.data() + r * cols_;
    }

    float& operator()(std::size_t r, std::size_t c) noexcept {
        assert(r < rows_ && c < cols_);
        return storage_[r * cols_ + c];
    }

    float operator()(std::size_t r, std::size_t c) const noexcept {
        assert(r < rows_ && c < cols_);
        return storage_[r * cols_ + c];
    }

    Tensor2DView view() const noexcept {
        return Tensor2DView(storage_.data(), rows_, cols_);
    }

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<float> storage_;
};

} // namespace vec_classify
