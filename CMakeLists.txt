cmake_minimum_required(VERSION 3.14)
project(esmc.cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp / ggml as subdirectory (LLAMA_STANDALONE=OFF disables tools/examples)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)

add_subdirectory(ggml)

add_library(esmc STATIC
    src/esmc.cpp
    src/esmc-graph.cpp
    src/esmc-tokenize.cpp
)
target_include_directories(esmc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(esmc PUBLIC ggml)
target_compile_features(esmc PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(esmc PUBLIC Threads::Threads)

add_executable(esmc-embed examples/embed/main.cpp)
target_link_libraries(esmc-embed PRIVATE esmc)
target_compile_features(esmc-embed PRIVATE cxx_std_17)

add_executable(esmc-bench examples/bench/main.cpp)
target_link_libraries(esmc-bench PRIVATE esmc)
target_compile_features(esmc-bench PRIVATE cxx_std_17)

add_executable(esmc-quantize examples/quantize/main.cpp)
target_link_libraries(esmc-quantize PRIVATE ggml)
target_compile_features(esmc-quantize PRIVATE cxx_std_17)

# Convenience: match plan verification `make -C build esmc-embed`
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
