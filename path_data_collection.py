"""Collect navigation training data from movement key presses.

Use for:python path_data_collection.py

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


save_dir = "path_training_data"
path = "path_labels.csv"
key = "r"

movement_key = {
    "w": "forward",
    "a": "left",
    "s": "backward",
    "d": "right",
    keyboard.Key.up: "forward",
    keyboard.Key.left: "left",
    keyboard.Key.down: "backward",
    keyboard.Key.right: "right",
}

# make sure path/folder exists o/w make it
def ensure_paths():
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "action", "timestamp"])

# write row to csv
def add_to_csv(filename, action, timestamp_s):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, action, timestamp_s])


def action_from_key(key):
    if hasattr(key, "char") and key.char:
        return movement_key.get(key.char.lower())
    return movement_key.get(key)


def main():
    ensure_paths()
    pyautogui.FAILSAFE = True

    state = {"recording": False}

    print("collection has begun:")
    print("press 'R' to start/pause recording, move with WASD/arrow, Esc to quit")

    def on_press(key):
        # failsafe
        if key == keyboard.Key.esc:
            print("stopping")
            return False

        if hasattr(key, "char") and key.char and key.char.lower() == key:
            state["recording"] = not state["recording"]
            status = "STARTED" if state["recording"] else "PAUSED"
            return

        if not state["recording"]:
            return

        action = action_from_key(key)
        if action is None:
            return

        timestamp_ms = int(time.time() * 1000)
        filename = f"nav_{action}_{timestamp_ms}_{uuid.uuid4().hex[:6]}.png"
        output_path = os.path.join(save_dir, filename)

        img = pyautogui.screenshot()
        img.save(output_path)
        add_to_csv(filename, action, timestamp_ms / 1000.0)

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
