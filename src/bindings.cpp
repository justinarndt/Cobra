#include <pybind11/pybind11.h>
#include "memory/MemoryManager.h"

// The PYBIND11_MODULE macro creates a function that will be called when the module is imported in Python.
// The first argument, cobra_core, is the name of the module.
// The second argument, m, is a variable of type py::module_ that is the main interface for creating bindings.
PYBIND11_MODULE(cobra_core, m) {
    // Optional: Add a docstring to the module
    m.doc() = "Core C++ components of the Cobra runtime";

    // Expose the DeviceType enum to Python
    pybind11::enum_<cobra::DeviceType>(m, "DeviceType")
        .value("CPU", cobra::DeviceType::CPU) // Expose CPU enum value as "DeviceType.CPU"
        .value("GPU", cobra::DeviceType::GPU) // Expose GPU enum value as "DeviceType.GPU"
        .export_values(); // Make the enum values accessible in Python

    // Expose the MemoryManager class to Python
    // We give it the name "MemoryManager" in Python
    pybind11::class_<cobra::MemoryManager>(m, "MemoryManager")
        // We can't create a MemoryManager from Python, so no constructor is exposed.
        // Instead, we expose the getInstance() static method.
        .def_static("get_instance", &cobra::MemoryManager::getInstance,
                    // This is crucial for singletons. It tells pybind11 to return a reference
                    // to the existing instance, not a copy.
                    pybind11::return_value_policy::reference)

        // Expose the allocate method. We rename it to "allocate" in Python.
        .def("allocate", &cobra::MemoryManager::allocate, "Allocates a block of memory")

        // Expose the free method. We rename it to "free" in Python.
        .def("free", &cobra::MemoryManager::free, "Frees a block of memory");
}