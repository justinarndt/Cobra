// src/memory/MemoryManager.cpp
//
// Implements the full logic for the MemoryManager class.

#include "memory/MemoryManager.h"
#include <iostream>

// Platform-specific headers for the CPUID instruction
#if defined(_MSC_VER)
#include <intrin.h> // For Visual Studio
#else
#include <cpuid.h>  // For GCC/Clang
#endif

// Initialize the static singleton pointer to nullptr.
MemoryManager* MemoryManager::s_instance = nullptr;

// ============================================================================
// Lifecycle Management Implementation
// ============================================================================

void MemoryManager::initialize() {
    if (!s_instance) {
        // This is the only place the constructor is ever called.
        s_instance = new MemoryManager();
    }
}

void MemoryManager::shutdown() {
    delete s_instance;
    s_instance = nullptr;
}

MemoryManager& MemoryManager::getInstance() {
    if (!s_instance) {
        // Enforce that cobra.init() must be called first.
        throw std::runtime_error("MemoryManager not initialized. Call cobra.init() first.");
    }
    return *s_instance;
}

// ============================================================================
// Constructor and Destructor (Runtime Setup/Teardown)
// ============================================================================

MemoryManager::MemoryManager() {
    std::cout << "C++: Initializing MemoryManager and Level Zero Runtime..." << std::endl;

    // Initialize member variables
    m_driver_handle = nullptr;
    m_device_handle = nullptr;
    m_context_handle = nullptr;
    m_command_queue = nullptr;
    m_command_list = nullptr;
    m_module_handle = nullptr;
    m_kernel_handle = nullptr;
    m_has_avx512 = false;
    m_has_amx = false;

    // 1. Initialize the Level Zero driver library
    zeInit(0);

    // 2. Discover the driver
    uint32_t driverCount = 1;
    zeDriverGet(&driverCount, &m_driver_handle);

    // 3. Discover the device (GPU)
    uint32_t deviceCount = 1;
    zeDeviceGet(m_driver_handle, &deviceCount, &m_device_handle);

    // 4. Create a context
    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    zeContextCreate(m_driver_handle, &context_desc, &m_context_handle);

    // 5. Create a command queue
    ze_command_queue_desc_t cmd_queue_desc = {};
    cmd_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cmd_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    zeCommandQueueCreate(m_context_handle, m_device_handle, &cmd_queue_desc, &m_command_queue);

    // 6. Create a command list
    ze_command_list_desc_t cmd_list_desc = {};
    cmd_list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    zeCommandListCreate(m_context_handle, m_device_handle, &cmd_list_desc, &m_command_list);

    // 7. Detect host CPU features
    detectCPUFeatures();

    std::cout << "C++: Level Zero Runtime Initialized Successfully." << std::endl;
}

MemoryManager::~MemoryManager() {
    std::cout << "C++: Shutting Down MemoryManager and Level Zero Runtime..." << std::endl;

    // Destroy resources in the reverse order of creation
    if (m_kernel_handle) zeKernelDestroy(m_kernel_handle);
    if (m_module_handle) zeModuleDestroy(m_module_handle);
    if (m_command_list) zeCommandListDestroy(m_command_list);
    if (m_command_queue) zeCommandQueueDestroy(m_command_queue);
    if (m_context_handle) zeContextDestroy(m_context_handle);

    std::cout << "C++: Level Zero Runtime Shut Down." << std.endl;
}

// ============================================================================
// Core Functionality Implementation
// ============================================================================

void* MemoryManager::allocate(size_t size) {
    void* ptr = nullptr;
    ze_device_mem_alloc_desc_t mem_desc = {};
    mem_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    ze_result_t result = zeMemAllocDevice(m_context_handle, &mem_desc, size, 1, m_device_handle, &ptr);
    if (result != ZE_RESULT_SUCCESS || !ptr) {
        throw std::runtime_error("Failed to allocate device memory.");
    }
    return ptr;
}

void MemoryManager::free(void* ptr) {
    if (ptr) {
        zeMemFree(m_context_handle, ptr);
    }
}

void MemoryManager::loadKernel(const std::vector<uint32_t>& spirv_binary, const char* kernelName) {
    // If a module/kernel is already loaded, destroy it first
    if (m_kernel_handle) {
        zeKernelDestroy(m_kernel_handle);
        m_kernel_handle = nullptr;
    }
    if (m_module_handle) {
        zeModuleDestroy(m_module_handle);
        m_module_handle = nullptr;
    }

    // Load the new SPIR-V module
    ze_module_desc_t module_desc = {};
    module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    module_desc.inputSize = spirv_binary.size() * sizeof(uint32_t);
    module_desc.pInputModule = reinterpret_cast<const uint8_t*>(spirv_binary.data());
    zeModuleCreate(m_context_handle, m_device_handle, &module_desc, &m_module_handle, nullptr);

    // Create the kernel from the loaded module
    ze_kernel_desc_t kernel_desc = {};
    kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernel_desc.pKernelName = kernelName;
    zeKernelCreate(m_module_handle, &kernel_desc, &m_kernel_handle);
}

void MemoryManager::launchKernel() {
    if (!m_kernel_handle) {
        throw std::runtime_error("No kernel loaded to launch.");
    }
    // A real implementation would set arguments and grid dimensions here.
    zeCommandListReset(m_command_list);
    // ... zeCommandListAppendLaunchKernel would go here ...
    zeCommandListClose(m_command_list);

    zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr);
    zeCommandQueueSynchronize(m_command_queue, UINT32_MAX);
}

// ============================================================================
// Private Helper Implementation
// ============================================================================

void MemoryManager::detectCPUFeatures() {
#if defined(_MSC_VER)
    int cpuInfo[4];
    // Check for AVX-512 Foundation instructions
    __cpuidex(cpuInfo, 7, 0);
    m_has_avx512 = (cpuInfo[1] & (1 << 16));
    // Check for AMX BF16 instructions
    __cpuidex(cpuInfo, 7, 1);
    m_has_amx = (cpuInfo[0] & (1 << 22));
#else // Assuming GCC/Clang
    unsigned int eax, ebx, ecx, edx;
    // Check for AVX-512 Foundation instructions
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        m_has_avx512 = (ebx & (1 << 16));
        // Check for AMX BF16 instructions
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        m_has_amx = (eax & (1 << 22));
    }
#endif

    if (m_has_avx512) std::cout << "C++: AVX-512 support detected." << std::endl;
    else std::cout << "C++: AVX-512 support not detected." << std::endl;

    if (m_has_amx) std::cout << "C++: AMX support detected." << std::endl;
    else std::cout << "C++: AMX support not detected." << std::endl;
}