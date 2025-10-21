# cobra/tools/profiler_cli.py
#
# Implements the command-line interface for `cobra-prof`. This tool allows
# a user to easily profile a Python script from the terminal.

import argparse
import runpy
from .. import profiler # Import the context manager we just created

def main():
    parser = argparse.ArgumentParser(
        description="Run a Python script with the Cobra profiler enabled."
    )
    parser.add_argument(
        "script_path",
        help="The path to the Python script to profile."
    )
    args = parser.parse_args()

    print(f"=== Running '{args.script_path}' with cobra-prof ===")
    
    with profiler.profile() as p:
        # runpy executes the script in the current process, allowing the
        # profiler's C++ hooks to capture the events.
        runpy.run_path(args.script_path, run_name="__main__")
    
    print("\n" + "="*50)
    p.print_report()
    print("="*50)

if __name__ == "__main__":
    main()