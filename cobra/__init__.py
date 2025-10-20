# cobra/__init__.py

# Expose the core, user-facing components of the Cobra library
# at the top-level package namespace.

from .stdlib import CobraArray
# This is the corrected import. It imports the 'jit' function
# from within the 'compiler.jit' module.
from .compiler.jit import jit

