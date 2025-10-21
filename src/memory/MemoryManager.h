// src/memory/MemoryManager.h
//
// Defines the interface for the core C++ MemoryManager singleton.
// This class is responsible for the entire lifecycle of the SYCL/Level Zero
// runtime, including device initialization, memory allocation, and kernel execution.

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <ze_api.h> // The oneAPI Level Zero header

class MemoryManager {
public:
    // ========================================================================
    // Lifecycle Management (The Deadlock Fix)
    // ========================================================================

    // Creates the singleton instance and initializes the Level Zero runtime.
    static void initialize();

    // Destroys the singleton instance and shuts down the runtime.
    static void shutdown();

    // Provides access to the singleton instance.
    // Throws an error if the manager has not been initialized.
    static MemoryManager& getInstance();

    // ========================================================================
    // Core Functionality
    // ========================================================================

    // Allocates a block of memory on the GPU device.
    void* allocate(size_t size);

    // Frees a block of device memory.
    void free(void* ptr);

    // Loads a SPIR-V binary onto the device, creating a runnable module.
    void loadKernel(const std::vector<uint32_t>& spirv_binary, const char* kernelName);

    // Launches the most recently loaded kernel.
    // (A real implementation would take more arguments for kernel args, grid size etc.)
    void launchKernel();


private:
    // The private static pointer that holds the single instance.
    static MemoryManager* s_instance;

    // The constructor is private to enforce the singleton pattern.
    // It performs the actual one-time setup of the Level Zero runtime.
    MemoryManager();

    // The destructor is private and handles the cleanup of all runtime resources.
    ~MemoryManager();

    // Disable copy and assign constructors to prevent multiple instances.
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // ========================================================================
    // Private Member Variables (Level Zero Handles)
    // ========================================================================
    // These handles represent the state of our connection to the hardware.

    ze_driver_handle_t m_driver_handle;
    ze_device_handle_t m_device_handle;
    ze_context_handle_t m_context_handle;
    ze_command_queue_handle_t m_command_queue;
    ze_command_list_handle_t m_command_list;
    ze_module_handle_t m_module_handle;
    ze_kernel_handle_t m_kernel_handle;
};

#endif // MEMORY_MANAGER_H