#include "MemoryManager.h"
#include <iostream> // For console output (std::cout)

namespace cobra {

    MemoryManager& MemoryManager::getInstance() {
        static MemoryManager instance; // The one and only instance
        return instance;
    }

    void* MemoryManager::allocate(size_t size, DeviceType device) {
        const char* deviceName = (device == DeviceType::GPU) ? "GPU" : "CPU";
        std::cout << "[MemoryManager] INFO: Requesting allocation of " << size
                  << " bytes on device: " << deviceName << std::endl;

        // In a real implementation, we would call malloc, cudaMalloc, etc.
        // For now, we return a null pointer as a placeholder.
        return nullptr;
    }

    void MemoryManager::free(void* ptr) {
        std::cout << "[MemoryManager] INFO: Requesting to free memory at address "
                  << ptr << std::endl;

        // In a real implementation, we would call free, cudaFree, etc.
    }

} // namespace cobra