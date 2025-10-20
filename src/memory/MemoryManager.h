#ifndef COBRA_MEMORY_MANAGER_H
#define COBRA_MEMORY_MANAGER_H

#include <iostream>
#include <stdexcept>
// Include the core SYCL header
#include <CL/sycl.hpp>

namespace cobra {

    // The DeviceType enum will now map directly to SYCL device selectors.
    enum class DeviceType {
        CPU,
        GPU
    };

    class MemoryManager {
    public:
        // Ensure the MemoryManager cannot be copied or moved.
        MemoryManager(const MemoryManager&) = delete;
        MemoryManager& operator=(const MemoryManager&) = delete;

        // Public static method to get the singleton instance.
        static MemoryManager& getInstance() {
            static MemoryManager instance; // Created once and only once.
            return instance;
        }

        // Allocates Unified Shared Memory (USM) on the selected device.
        // Returns a raw void pointer to the allocated memory.
        void* allocate(size_t bytes, DeviceType device) {
            sycl::queue& q = get_queue(device);
            void* ptr = sycl::malloc_shared(bytes, q);
            if (!ptr) {
                throw std::runtime_error("Failed to allocate USM memory.");
            }
            std::cout << "[MemoryManager] INFO: Allocated " << bytes << " bytes of USM on device: "
                      << q.get_device().get_info<sycl::info::device::name>() << std::endl;
            return ptr;
        }

        // Frees Unified Shared Memory.
        void free(void* ptr, DeviceType device) {
            if (ptr) {
                sycl::queue& q = get_queue(device);
                sycl::free(ptr, q);
                 std::cout << "[MemoryManager] INFO: Freed USM memory at address " << ptr << std::endl;
            }
        }

    private:
        // Private constructor to enforce the singleton pattern.
        MemoryManager() {
            try {
                // Initialize queues for available devices upon creation.
                cpu_queue = sycl::queue(sycl::cpu_selector_v);
                 std::cout << "[MemoryManager] INFO: Initialized SYCL queue for CPU: "
                          << cpu_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
            } catch (const sycl::exception& e) {
                 std::cerr << "[MemoryManager] WARNING: Could not initialize SYCL CPU queue: " << e.what() << std::endl;
            }
            try {
                gpu_queue = sycl::queue(sycl::gpu_selector_v);
                 std::cout << "[MemoryManager] INFO: Initialized SYCL queue for GPU: "
                          << gpu_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
            } catch (const sycl::exception& e) {
                 std::cerr << "[MemoryManager] WARNING: Could not initialize SYCL GPU queue. This is expected if no discrete GPU is present. " << e.what() << std::endl;
            }
        }

        // Returns the appropriate SYCL queue for the given device type.
        sycl::queue& get_queue(DeviceType device) {
            if (device == DeviceType::GPU && gpu_queue.has_value()) {
                return gpu_queue.value();
            }
            // Default to CPU if GPU is requested but not available.
            return cpu_queue.value();
        }

        // SYCL queues for CPU and GPU devices.
        // std::optional is used because a GPU might not be present.
        std::optional<sycl::queue> cpu_queue;
        std::optional<sycl::queue> gpu_queue;
    };

} // namespace cobra

#endif // COBRA_MEMORY_MANAGER_H