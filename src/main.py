# main.py
import importlib
import subprocess
import os

# Days with Streamlit apps
streamlit_days = {2, 4, 8, 9, 10}

def main():
    day = input(" Enter the day number you want to run (1–10): ").strip()

    if not day.isdigit() or not (1 <= int(day) <= 10):
        print(" Please enter a valid day number (1–10).")
        return

    day = int(day)

    if day in streamlit_days:
        print(f"📦 Launching Streamlit app for Day {day}...")
        subprocess.run(["streamlit", "run", f"day{day}.py"], cwd=os.path.dirname(__file__))
    else:
        try:
            module = importlib.import_module(f"day{day}")
            if hasattr(module, f"run_day{day}"):
                print(f"🚀 Running Day {day} task...")
                getattr(module, f"run_day{day}")()
            else:
                print(f"No 'run_day{day}()' function found in day{day}.py")
        except ModuleNotFoundError:
            print(f" File day{day}.py not found.")

if __name__ == "__main__":
    main()
