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
                r, g, b = screen.getpixel((x, y))

                # Logic: Is it 'Green enough'? 
                # (High green, low red, low blue)
                if g > MIN_GREEN and r < MAX_RED and b < MAX_BLUE:
                    print(f"Target shade found at ({x}, {y})! Clicking...")
                    
                    # Move and Click
                    pyautogui.click(x, y)
                    
                    # 4. THE WAIT
                    # Wait 6 seconds for the character to walk over and chop.
                    # Then click again in the same spot to 'Loot' the log!
                    time.sleep(6)
                    print("Attempting to loot log...")
                    pyautogui.click(x, y) 
                    
                    found_target = True
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