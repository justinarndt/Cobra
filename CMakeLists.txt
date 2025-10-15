# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Cobra)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Cobra C++ Core Library ---
# Create a library for our core runtime.
add_library(cobra_runtime STATIC
    src/memory/MemoryManager.cpp
)

# Tell CMake that our library's header files are located in the /src directory.
target_include_directories(cobra_runtime PUBLIC ${PROJECT_SOURCE_DIR}/src)


# --- Python Bindings ---
# Find the pybind11 package that we added as a submodule
add_subdirectory(tools/pybind11)

# pybind11_add_module is a special function provided by pybind11.
# It creates a Python module that is correctly configured for the current OS.
# The first argument is the name of the module target: cobra_core
# The second argument is the source file for the bindings.
pybind11_add_module(cobra_core SHARED
    src/bindings.cpp
)

# Link our Python module with our cobra_runtime C++ library.
# This makes the MemoryManager code available to the bindings.
target_link_libraries(cobra_core PRIVATE cobra_runtime)