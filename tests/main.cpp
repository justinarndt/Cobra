#include "memory/MemoryManager.h" // Include our memory manager header
#include <iostream>

int main() {
    std::cout << "--- Starting Cobra Test Runner ---" << std::endl;

    // Get the singleton instance of the MemoryManager
    cobra::MemoryManager& mm = cobra::MemoryManager::getInstance();

    // Use the memory manager to perform placeholder allocations
    void* cpu_ptr = mm.allocate(1024, cobra::DeviceType::CPU);
    void* gpu_ptr = mm.allocate(4096, cobra::DeviceType::GPU);

    // Use the memory manager to free the placeholder allocations
    mm.free(cpu_ptr);
    mm.free(gpu_ptr);

    std::cout << "--- Cobra Test Runner Finished ---" << std::endl;

    return 0; // Exit with success code
}