import pyautogui
import time

# 1. THE PANIC BUTTON
# If the mouse goes crazy, slam your cursor into the TOP-LEFT corner of your screen!
pyautogui.FAILSAFE = True

def run_color_test():
    # 2. TARGET SETTINGS (Shades of Green)
    # These ranges are for the 'Average' green of a Beech Tree.
    # Adjust these based on what you see in your Digital Color Meter!
    MAX_RED = 130
    MIN_GREEN = 50
    MAX_BLUE = 135

    print("--- STEINNET TEST ONLINE ---")
    print("You have 5 seconds to switch to Steinworld...")
    time.sleep(5)

    while True:
        # Take a snapshot of the screen
        screen = pyautogui.screenshot()
        width, height = screen.size

        found_target = False

        # 3. THE SCANNER
        # We scan every 20th pixel to save your Mac's CPU power
        for x in range(0, width, 20):
            for y in range(0, height, 20):
                r, g, b = screen.getpixel((x, y))[:3]

                # Logic: Is it 'Green enough'? 
                # (High green, low red, low blue)
                if g > MIN_GREEN and r < MAX_RED and b < MAX_BLUE:                     
                    found_target = True
                    # CALIBRATE FOR MAC: Divide by 2
                    target_x = x / 2
                    target_y = y / 2
    
                    print(f"Found pixels at {x},{y}. Moving mouse to logical {target_x},{target_y}")
    
                    # Move and Click with a tiny duration so you can see it move
                    pyautogui.moveTo(target_x, target_y, duration=0.2)
                    pyautogui.click()
    
                    time.sleep(6) # Wait for chopping
                    pyautogui.click() # Looting click
                    break
            if found_target: break

        # Short pause before looking for the next tree
        time.sleep(2)

# THE IGNITION
if __name__ == "__main__":
    try:
        run_color_test()
    except pyautogui.FailSafeException:
        print("\nFAIL-SAFE TRIGGERED. Script stopped.")