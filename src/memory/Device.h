#ifndef COBRA_DEVICE_H
#define COBRA_DEVICE_H

namespace cobra {

    // Enum to represent the physical device where memory can be allocated.
    enum class DeviceType {
        CPU,
        GPU
    };

} // namespace cobra

#endif //COBRA_DEVICE_H