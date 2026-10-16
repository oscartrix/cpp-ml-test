cmake_minimum_required(VERSION 3.14)
project(vec_classify LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_ASAN "Compilar con AddressSanitizer y UndefinedBehaviorSanitizer" OFF)

add_compile_options(-Wall -Wextra)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

add_library(vec_classify_lib
    src/dataset.cpp
    src/distance.cpp
    src/index.cpp
    src/classifier.cpp
)
target_include_directories(vec_classify_lib PUBLIC include)

# alloc_hook.cpp reemplaza ::operator new y se compila en cada ejecutable (no en la librería)
add_executable(benchmark_cli src/main_benchmark.cpp src/alloc_hook.cpp)
target_link_libraries(benchmark_cli PRIVATE vec_classify_lib)

enable_testing()
add_executable(test_suite tests/test_starter_audit.cpp src/alloc_hook.cpp)
target_link_libraries(test_suite PRIVATE vec_classify_lib)
add_test(NAME test_suite COMMAND test_suite)
