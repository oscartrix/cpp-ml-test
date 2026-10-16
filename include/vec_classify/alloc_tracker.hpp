#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace vec_classify {

/**
 * Rastreador de alocaciones de memoria dinámica por hilo (Thread-Local).
 * Permite verificar que el camino crítico de inferencia (`search()`) opere con
 * CERO alocaciones en el heap cuando se usa un buffer `SearchResult` pre-reservado.
 */
class ScopedAllocationTracker {
public:
    ScopedAllocationTracker() {
        prev_state_ = tracking_enabled();
        alloc_count() = 0;
        tracking_enabled() = true;
    }

    ~ScopedAllocationTracker() {
        tracking_enabled() = prev_state_;
    }

    static std::size_t current_allocations() noexcept {
        return alloc_count();
    }

    static void record_allocation() noexcept {
        if (tracking_enabled()) {
            ++alloc_count();
        }
    }

private:
    bool prev_state_{false};

    static bool& tracking_enabled() noexcept {
        thread_local bool enabled = false;
        return enabled;
    }

    static std::size_t& alloc_count() noexcept {
        thread_local std::size_t count = 0;
        return count;
    }
};

} // namespace vec_classify
