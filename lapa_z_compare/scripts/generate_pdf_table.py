import sys
from pathlib import Path

# Add script directory to path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from compile_results import main as compile_main

def main():
    compile_main()

if __name__ == "__main__":
    main()
