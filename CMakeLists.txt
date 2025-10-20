# Set the minimum required version of CMake.
cmake_minimum_required(VERSION 3.10)

# Set the project name.
project(Cobra)

# --- Intel oneAPI / SYCL Integration ---
# Find the SYCL package provided by the oneAPI DPC++/C++ Compiler installation.
# This is the crucial step that locates the necessary headers and libraries.
# It will define the SYCL_INCLUDE_DIRS and SYCL_LIBRARIES variables.
find_package(SYCL REQUIRED)
# ---------------------------------------

# Add the pybind11 submodule, which we use for C++/Python bindings.
add_subdirectory(tools/pybind11)

# --- Cobra Runtime Library (C++) ---
# This static library contains the core C++ logic, like the MemoryManager.
add_library(cobra_runtime src/memory/MemoryManager.cpp)

# Add the necessary include directories for the runtime library.
# We now add ${SYCL_INCLUDE_DIRS} to this list.
target_include_directories(cobra_runtime PUBLIC
    ${pybind11_INCLUDE_DIRS}
    ${SYCL_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/src" # Include the top-level src directory
)

# Link the runtime library against the SYCL runtime libraries.
target_link_libraries(cobra_runtime PUBLIC ${SYCL_LIBRARIES})

# --- Cobra Core Python Module (Bindings) ---
# This target uses pybind11 to create the final .pyd/.so file
# that Python will import.
pybind11_add_module(cobra_core src/bindings.cpp)

# Link the Python module against our internal C++ runtime library.
target_link_libraries(cobra_core PRIVATE cobra_runtime)
