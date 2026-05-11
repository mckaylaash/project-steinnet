import pyautogui
from pynput import mouse, keyboard
import uuid
import os


path = "training_data"
if not os.path.exists(path):
    os.makedirs(path)

# adjust dimensions to Mac screen size
logical_w, logical_h = pyautogui.size()
SCALE = pyautogui.screenshot().width / logical_w

recording_active = False

print("press 's' to start, 'q' to quit.")

def on_click(x, y, button, pressed):
    ''' 
    Mouse click handler that captures the screen and saves a screenshot with the clicked coordinates encoded in the filename. It only activates when recording_active is True and the left mouse button is clicked.
    filename format is target_X_Y_uuid.png, where X and Y are the pixel coordinates of the click, and uuid is a random string for uniqueness.
    
    '''
    global recording_active
    if recording_active and pressed and button == mouse.Button.left:
        img = pyautogui.screenshot()
        pixel_x, pixel_y = int(x * SCALE), int(y * SCALE)
        
        filename = f"{path}/target_{pixel_x}_{pixel_y}_{uuid.uuid4().hex[:6]}.png"
        img.save(filename)
        print(f"Captured: {pixel_x}, {pixel_y}")

def on_press(key):
    global recording_active
    try:
        if key.char == 's':
            recording_active = True
        elif key.char == 'q':
            print("end")
            return False 
    except AttributeError:
        pass

# start keyboard and mouse "listener", or ability to capture
with mouse.Listener(on_click=on_click) as m_listener:
    with keyboard.Listener(on_press=on_press) as k_listener:
        k_listener.join()
        m_listener.stop()
