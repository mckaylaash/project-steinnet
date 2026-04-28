import pyautogui
import time
import keyboard  # New library for the key-kill

# 1. THE PANIC BUTTONS
pyautogui.FAILSAFE = True

def run_improved_test():
    logical_w, logical_h = pyautogui.size()
    screenshot = pyautogui.screenshot()
    scale = screenshot.width / logical_w
    
    # 2. SEARCH ZONE (Around your character in the center)
    center_x, center_y = (screenshot.width / 2), (screenshot.height / 2)
    zone_size = 400  # Increased this slightly so it's easier to find stuff
    
    print("--- SCRIPT ACTIVE ---")
    print("Hold 'q' at any time to ABORT.")
    time.sleep(5)

    while True:
        # NEW: Check for the 'q' key every loop
        if keyboard.is_pressed('q'):
            print("Keyboard interrupt! Stopping...")
            break

        screen = pyautogui.screenshot()
        found_target = False

        # 3. RELAXED COLOR SEARCH
        # If it wasn't moving, the previous green was too strict.
        # Let's look for anything that is 'Mostly Green'
        for x in range(int(center_x - zone_size), int(center_x + zone_size), 15):
            for y in range(int(center_y - zone_size), int(center_y + zone_size), 15):
                
                # Safety check to stay within screen bounds
                if x >= screen.width or y >= screen.height: continue
                
                r, g, b = screen.getpixel((x, y))[:3]

                # RELAXED LOGIC: Green must be the dominant color
                if g > r and g > b and g > 80:
                    target_x = x / scale
                    target_y = y / scale
                    
                    print(f"Green spotted! Moving to {target_x}, {target_y}")
                    pyautogui.moveTo(target_x, target_y, duration=0.2)
                    pyautogui.click()
                    
                    time.sleep(6) # Wait for chopping
                    pyautogui.click() # Looting click
                    found_target = True
                    break
            if found_target: break

if __name__ == "__main__":
    run_improved_test()