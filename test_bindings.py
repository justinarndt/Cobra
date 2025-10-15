print("--- Starting Python Bindings Test ---")

try:
    # We expect this to fail initially
    import cobra_core
except ImportError:
    print("Initial import failed as expected. Copying module and retrying...")
    import os
    import shutil
    
    # On Windows, pybind11 might add a suffix like 'cp312-win_amd64'
    # We need to find the actual file name.
    build_dir = os.path.join('build', 'Debug')
    found_module = None
    for f in os.listdir(build_dir):
        if f.startswith('cobra_core') and f.endswith('.pyd'):
            found_module = os.path.join(build_dir, f)
            break
            
    if found_module:
        print(f"Found module at: {found_module}")
        # Copy the file to the current directory so Python can find it
        shutil.copy(found_module, 'cobra_core.pyd')
        print("Copied module to 'cobra_core.pyd' in current directory.")
        # Now the import should work
        import cobra_core
    else:
        print("Error: Compiled module 'cobra_core.*.pyd' not found in build/Debug!")
        exit()


# Get the MemoryManager singleton instance from our C++ code
mm = cobra_core.MemoryManager.get_instance()
print(f"Successfully got MemoryManager instance: {mm}")

# Access the DeviceType enum from C++
CPU = cobra_core.DeviceType.CPU
GPU = cobra_core.DeviceType.GPU
print(f"Successfully accessed enums: {CPU}, {GPU}")

# Call the allocate and free methods, which will trigger the C++ cout messages
print("\nCalling C++ functions from Python...")
cpu_ptr = mm.allocate(2048, CPU)
gpu_ptr = mm.allocate(8192, GPU)
mm.free(cpu_ptr)
mm.free(gpu_ptr)

print("\n--- Python Bindings Test Finished ---")