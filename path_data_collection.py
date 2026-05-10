"""Collect navigation training data from movement key presses.

Usage:
  python path_data_collection.py

Controls:
  - Press R to start/pause recording
  - Press ESC to quit
  - While recording, every W/A/S/D or Arrow key press captures a screenshot
"""

import csv
import os
import time
import uuid

import pyautogui
from pynput import keyboard


SAVE_DIR = "path_training_data"
LOG_PATH = "path_labels.csv"
TOGGLE_KEY = "r"

ACTION_KEYS = {
    "w": "forward",
    "a": "left",
    "s": "backward",
    "d": "right",
    keyboard.Key.up: "forward",
    keyboard.Key.left: "left",
    keyboard.Key.down: "backward",
    keyboard.Key.right: "right",
}


def ensure_paths():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "action", "timestamp"])


def append_log_row(filename, action, timestamp_s):
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, action, timestamp_s])


def action_from_key(key):
    if hasattr(key, "char") and key.char:
        return ACTION_KEYS.get(key.char.lower())
    return ACTION_KEYS.get(key)


def main():
    ensure_paths()
    pyautogui.FAILSAFE = True

    state = {"recording": False}

    print("--- PATH DATA COLLECTION ---")
    print("1. Switch to Steinworld.")
    print("2. Press 'R' to START/PAUSE recording.")
    print("3. While recording, move with WASD or Arrow keys.")
    print("4. Press ESC to quit.")

    def on_press(key):
        # Quit at any time.
        if key == keyboard.Key.esc:
            print("Stopping collector.")
            return False

        # Toggle recording mode.
        if hasattr(key, "char") and key.char and key.char.lower() == TOGGLE_KEY:
            state["recording"] = not state["recording"]
            status = "STARTED" if state["recording"] else "PAUSED"
            print(f">> RECORDING {status}")
            return

        if not state["recording"]:
            return

        action = action_from_key(key)
        if action is None:
            return

        timestamp_ms = int(time.time() * 1000)
        filename = f"nav_{action}_{timestamp_ms}_{uuid.uuid4().hex[:6]}.png"
        output_path = os.path.join(SAVE_DIR, filename)

        img = pyautogui.screenshot()
        img.save(output_path)
        append_log_row(filename, action, timestamp_ms / 1000.0)
        print(f"Captured: {filename} -> {action}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
