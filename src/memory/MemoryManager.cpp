#include "MemoryManager.h"
#include <iostream>

namespace cobra {

    MemoryManager& MemoryManager::getInstance() {
        static MemoryManager instance;
        return instance;
    }

    void* MemoryManager::allocate(size_t size, DeviceType device) {
        const char* deviceName = (device == DeviceType::GPU) ? "GPU" : "CPU";
        std::cout << "[MemoryManager] INFO: Requesting allocation of " << size
                  << " bytes on device: " << deviceName << std::endl;

        // --- CHANGE ---
        // Instead of returning nullptr, we now allocate a single dummy byte
        // on the C++ heap. This gives us a REAL, non-null pointer to use
        // as a placeholder handle. We will use the size of the requested
        // allocation to make the handle unique for this test.
        char* dummy_handle = new char[size];
        std::cout << "[MemoryManager] DEBUG: Created dummy handle at address "
                  << static_cast<void*>(dummy_handle) << std::endl;
        return dummy_handle;
    }

    void MemoryManager::free(void* ptr) {
        std::cout << "[MemoryManager] INFO: Requesting to free memory at address "
                  << ptr << std::endl;

        // --- CHANGE ---
        // Now that we are allocating real C++ memory, we must free it to
        // prevent memory leaks. The 'delete[]' operator matches the 'new char[]'
        // in the allocate function.
        delete[] static_cast<char*>(ptr);
    }

} // namespace cobra