# cobra/profiler.py
#
# Defines the Python-side logic for the profiler, providing a clean
# API and context manager for profiling code blocks.

from .runtime import cobra_core

class profile:
    """
    A context manager for profiling a block of Cobra code.
    
    Example:
        with cobra.profile() as p:
            # Code to profile...
        p.print_report()
    """
    def __enter__(self):
        cobra_core.profiler_clear_events()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.events = cobra_core.profiler_get_events()

    def print_report(self):
        print("--- Cobra Profiler Report ---")
        if not self.events:
            print("No events captured.")
            return
            
        total_time = sum(duration for _, duration in self.events)
        
        print(f"Total GPU Time: {total_time:.4f} ms")
        print("-----------------------------")
        
        for name, duration in self.events:
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{name:<25} | {duration:>10.4f} ms | ({percentage:5.1f}%)")
        print("-----------------------------")