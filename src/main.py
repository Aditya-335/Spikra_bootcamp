import argparse
import importlib
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run bootcamp day")
    parser.add_argument("--day", type=int, required=True, help="Day number to run (1–10)")
    args = parser.parse_args()

    day = args.day
    if day == 2:
        print("🚀 Launching Streamlit app for Day 2...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/day2.py"])
        return

    try:
        module = importlib.import_module(f"day{day}")
        func_name = f"run_day{day}"
        if hasattr(module, func_name):
            getattr(module, func_name)()
        else:
            print(f"❌ '{func_name}()' not found in day{day}.py")
    except ModuleNotFoundError:
        print(f"❌ day{day}.py not found in src/")

if __name__ == "__main__":
    main()
