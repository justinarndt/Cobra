// MODIFIED: src/memory/MemoryManager.cpp
#include <ze_api.h> // Include the Level Zero header

// ... inside the MemoryManager class...

// This function takes a SPIR-V binary and loads it onto the target GPU
// device using the oneAPI Level Zero API.
void MemoryManager::loadKernel(const std::vector<uint32_t>& spirv_binary) {
    ze_module_desc_t module_desc = {};
    module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    module_desc.inputSize = spirv_binary.size() * sizeof(uint32_t);
    module_desc.pInputModule = reinterpret_cast<const uint8_t*>(spirv_binary.data());

    ze_module_handle_t module_handle;
    ze_result_t result = zeModuleCreate(
        m_sycl_device_handle, // Assumes we have these from cobra.init()
        m_sycl_context_handle,
        &module_desc,
        &module_handle,
        nullptr);

    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Level Zero: Failed to create module from SPIR-V.");
    }

    // Store the module_handle for later use in kernel creation and launch.
    m_module_handle = module_handle;
}