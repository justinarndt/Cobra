# src/CMakeLists.txt
#
# CMake configuration for the core C++ library.

# Define the library target for our C++ code.
add_library(cobra_core SHARED
    bindings.cpp
    memory/MemoryManager.cpp
    # Add other .cpp files here as they are created
)

# Tell the target where to find the pybind11 headers.
target_include_directories(cobra_core PRIVATE
    ../tools/pybind11/include
)

# --- Use the oneAPI paths found in the root CMakeLists.txt ---
# Add the Level Zero header path to the include directories.
target_include_directories(cobra_core PRIVATE
    ${ONEAPI_LEVEL_ZERO_INCLUDE_DIR}
)

# Link the target against the Level Zero loader library.
target_link_libraries(cobra_core PRIVATE
    ${ONEAPI_LEVEL_ZERO_LIBRARY}
)

# Set the proper suffix for a Python extension module (.pyd on Windows)
set_target_properties(cobra_core PROPERTIES PREFIX "" SUFFIX ".pyd")