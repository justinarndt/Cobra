# cobra/patcher.py
#
# Implements the dynamic, runtime patching (monkey-patching) functionality
# to accelerate third-party libraries.

import importlib
import sys

_patched_libs = set()

def patch_sklearn():
    """
    Dynamically replaces stock scikit-learn estimators with their
    Cobra-accelerated counterparts.

    This function should be called once, early in a user's script, before
    the target modules are imported.
    """
    global _patched_libs
    if "sklearn" in _patched_libs:
        print("Cobra: Scikit-learn has already been patched.")
        return

    print("Cobra: Patching Scikit-learn with accelerated components...")

    # The target module we want to patch.
    target_module_name = "sklearn.cluster"

    # Our module containing the replacement classes.
    patch_module_name = "cobra.sklearn.cluster"

    # Import our module containing the accelerated KMeans class.
    patch_module = importlib.import_module(patch_module_name)

    # If the user has already imported sklearn.cluster, we need to get a
    # reference to it. Otherwise, we create a mock module to hold the patch.
    try:
        target_module = sys.modules[target_module_name]
    except KeyError:
        # If not imported yet, create a placeholder. When the user later
        # imports it, Python will find this existing module in sys.modules.
        target_module = importlib.util.module_from_spec(
            importlib.util.find_spec(target_module_name)
        )
        sys.modules[target_module_name] = target_module

    # This is the "monkey-patch": we overwrite the `KMeans` attribute in
    # the original scikit-learn module namespace with our own class.
    setattr(target_module, 'KMeans', patch_module.PatchedKMeans)

    _patched_libs.add("sklearn")
    print("Cobra: Scikit-learn patching complete.")