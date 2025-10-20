#ifndef COBRA_MEMORY_MANAGER_H
#define COBRA_MEMORY_MANAGER_H

#include <stdexcept>
#include <optional>
#include <mutex>

#include <sycl/sycl.hpp>

namespace cobra {

    enum class DeviceType {
        CPU,
        GPU
    };

    class MemoryManager {
    public:
        MemoryManager(const MemoryManager&) = delete;
        MemoryManager& operator=(const MemoryManager&) = delete;

        static MemoryManager& getInstance() {
            static MemoryManager instance;
            return instance;
        }

        // --- THE WARM-UP FIX ---
        // This new public method will be called explicitly from Python to safely
        // trigger the one-time SYCL runtime initialization.
        void warm_up() {
            get_queue(DeviceType::CPU);
        }

        void* allocate(size_t bytes, DeviceType device) {
            sycl::queue& q = get_queue(device);
            void* ptr = sycl::malloc_shared(bytes, q);
            if (!ptr) {
                throw std::runtime_error("Failed to allocate USM memory.");
            }
            return ptr;
        }

        void free(void* ptr, DeviceType device) {
            if (ptr) {
                sycl::queue& q = get_queue(device);
                sycl::free(ptr, q);
            }
        }

    private:
        // The constructor remains empty. No work is done on DLL load.
        MemoryManager() = default;

        // This method performs the one-time initialization of SYCL queues.
        void initialize_queues() {
            try {
                cpu_queue.emplace(sycl::cpu_selector_v);
            } catch (const sycl::exception&) {
                // Suppress errors if a device is not found.
            }
            try {
                gpu_queue.emplace(sycl::gpu_selector_v);
            } catch (const sycl::exception&) {
                // Suppress errors if a device is not found.
            }
        }

        // Returns the appropriate SYCL queue, initializing on first call.
        sycl::queue& get_queue(DeviceType device) {
            // Use a thread-safe, one-time call to initialize the queues.
            std::call_once(init_flag, &MemoryManager::initialize_queues, this);

            if (device == DeviceType::GPU && gpu_queue.has_value()) {
                return gpu_queue.value();
            }
            if (cpu_queue.has_value()){
                return cpu_queue.value();
            }
            throw std::runtime_error("No valid SYCL queue available.");
        }

        std::optional<sycl::queue> cpu_queue;
        std::optional<sycl::queue> gpu_queue;
        std::once_flag init_flag;
    };

} // namespace cobra

#endif // COBRA_MEMORY_MANAGER_H