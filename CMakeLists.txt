# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Cobra)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create a library for our core runtime.
# For now, it's a STATIC library, meaning its code will be copied
# directly into whatever program uses it.
add_library(cobra_runtime STATIC
    src/memory/MemoryManager.cpp
)

# Tell CMake that our library's header files are located in the /src directory.
# This allows our .cpp files to find them using #include "..."
target_include_directories(cobra_runtime PUBLIC ${PROJECT_SOURCE_DIR}/src)

# --- Test Harness ---
# Define a new target, this time an executable program
add_executable(cobra_test_runner
    tests/main.cpp
)

# Link our test runner executable with our cobra_runtime library.
# This makes the MemoryManager code available to our test runner.
target_link_libraries(cobra_test_runner PRIVATE cobra_runtime)