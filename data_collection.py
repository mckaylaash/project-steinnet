import pyautogui
from pynput import mouse, keyboard
import uuid
import os


SAVE_PATH = "training_data"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Auto-calibrate Retina
logical_w, logical_h = pyautogui.size()
SCALE = pyautogui.screenshot().width / logical_w

# State control
recording_active = False

print(f"--- DATA COLLECTION (Scale: {SCALE}) ---")
print("1. Switch to Steinworld.")
print("2. Position your character.")
print("3. Press 's' to START recording clicks.")
print("4. Press 'q' to STOP and exit.")

def on_click(x, y, button, pressed):
    global recording_active
    if recording_active and pressed and button == mouse.Button.left:
        img = pyautogui.screenshot()
        pixel_x, pixel_y = int(x * SCALE), int(y * SCALE)
        
        filename = f"{SAVE_PATH}/target_{pixel_x}_{pixel_y}_{uuid.uuid4().hex[:6]}.png"
        img.save(filename)
        print(f"Captured: {pixel_x}, {pixel_y}")

def on_press(key):
    global recording_active
    try:
        if key.char == 's':
            recording_active = True
            print(">> RECORDING STARTED. Go chop some trees!")
        elif key.char == 'q':
            print(">> STOPPING...")
            return False # Stops the listener
    except AttributeError:
        pass

# Start both listeners
with mouse.Listener(on_click=on_click) as m_listener:
    with keyboard.Listener(on_press=on_press) as k_listener:
        k_listener.join()
        m_listener.stop()