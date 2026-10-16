#include "vec_classify/alloc_tracker.hpp"

#include <cstdlib>
#include <new>

// Hooks globales de ::operator new para contabilizar alocaciones en el hot-path.
// Se compilan directamente en cada ejecutable (no en la librería estática) para que el
// linker los incluya siempre, aunque el ejecutable no use ningún otro símbolo de la librería.

namespace {

void* tracked_malloc(std::size_t size) noexcept {
    vec_classify::ScopedAllocationTracker::record_allocation();
    return std::malloc(size == 0 ? 1 : size);
}

void* tracked_aligned_alloc(std::size_t size, std::align_val_t al) noexcept {
    vec_classify::ScopedAllocationTracker::record_allocation();
    const std::size_t align = static_cast<std::size_t>(al);
    const std::size_t rounded = ((size == 0 ? 1 : size) + align - 1) / align * align;
    return std::aligned_alloc(align, rounded);
}

} // namespace

void* operator new(std::size_t size) {
    if (void* ptr = tracked_malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return tracked_malloc(size);
}

void* operator new(std::size_t size, std::align_val_t al) {
    if (void* ptr = tracked_aligned_alloc(size, al)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    return tracked_aligned_alloc(size, al);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::align_val_t) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
    std::free(ptr);
}
