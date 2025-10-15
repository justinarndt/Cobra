#ifndef COBRA_MEMORYMANAGER_H
#define COBRA_MEMORYMANAGER_H

#include <cstddef> // Required for size_t
#include "Device.h"

namespace cobra {

    class MemoryManager {
    public:
        // Deleted copy constructor and assignment operator to prevent duplication
        MemoryManager(const MemoryManager&) = delete;
        MemoryManager& operator=(const MemoryManager&) = delete;

        // Provides the single global instance of the MemoryManager
        static MemoryManager& getInstance();

        // Allocates a block of memory of a given size on a specified device
        void* allocate(size_t size, DeviceType device);

        // Frees a previously allocated block of memory
        void free(void* ptr);

    private:
        // Private constructor to enforce the Singleton pattern
        MemoryManager() = default;
    };

} // namespace cobra

#endif //COBRA_MEMORYMANAGER_H