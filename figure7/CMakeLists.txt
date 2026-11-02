cmake_minimum_required(VERSION 3.16)

# The baselines' mix drivers, odinmix (OdinANN) and condamix (CONDA) over PipeANN, the pipeann submodule.
#
# Each driver compiles in PipeANN's directory scope (definitions, options, includes): those scopes
# carry system-specific definitions (-DUSE_URING/-DBG_IO_THREAD, the AVX probes) which change struct
# layout, so a translation unit compiled outside the scope it links against would be ABI-mismatched.

project(fncteval CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "build type")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/.local)

# 0015/fnctbench.hpp links benchlog, the Rust static library in 0015/benchlog with the bench's job
# shuffle and Arrow log.
find_package(OpenMP REQUIRED)
find_program(CARGO cargo HINTS $ENV{HOME}/.cargo/bin REQUIRED)
set(BENCHLOG "${CMAKE_CURRENT_SOURCE_DIR}/target/benchlog/release/libbenchlog.a")
add_custom_command(OUTPUT "${BENCHLOG}"
  COMMAND "${CARGO}" build --release --manifest-path "0015/benchlog/Cargo.toml"
    --target-dir target/benchlog
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "0015/benchlog/Cargo.toml" "0015/benchlog/lib.rs"
  VERBATIM)
add_custom_target(benchlog DEPENDS "${BENCHLOG}")

# PipeANN does not put its BLAS provider on the path: its library link line hardcodes -lcblas, and
# MKL's headers sit under a mkl/ suffix. Both propagate to the subdirectory added below.
find_path(MKLINCDIR mkl_cblas.h PATH_SUFFIXES mkl REQUIRED)
find_library(CBLASLIB NAMES cblas HINTS $ENV{HOME}/.local/lib REQUIRED)
get_filename_component(CBLASDIR ${CBLASLIB} DIRECTORY)
include_directories(${MKLINCDIR})
link_directories(${CBLASDIR})
# PipeANN's background writers, upstream's one.
set(PIPEANN_BG_IO_THREADS 1 CACHE STRING "PipeANN background IO threads")
add_compile_definitions(PIPEANN_BG_IO_THREADS=${PIPEANN_BG_IO_THREADS})
add_subdirectory(pipeann)

get_directory_property(PIPEDEFS DIRECTORY pipeann COMPILE_DEFINITIONS)
get_directory_property(PIPEOPTS DIRECTORY pipeann COMPILE_OPTIONS)
get_directory_property(PIPEINCS DIRECTORY pipeann INCLUDE_DIRECTORIES)
add_executable(odinmix "0015/odinmix.cc")
add_executable(condamix "0015/condamix.cc")
foreach(odin odinmix condamix)
  add_dependencies(${odin} benchlog)
  target_include_directories(${odin} PRIVATE "0015" ${PIPEINCS})
  target_compile_definitions(${odin} PRIVATE ${PIPEDEFS})
  target_compile_options(${odin} PRIVATE ${PIPEOPTS})
  target_link_libraries(${odin} PRIVATE pipeann "${BENCHLOG}" dl)
  set_target_properties(${odin} PROPERTIES CXX_STANDARD 20)
endforeach()
