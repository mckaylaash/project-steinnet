import pyautogui
import time

# 1. THE PANIC BUTTON
pyautogui.FAILSAFE = True

def run_calibrated_test():
    # --- AUTO-CALIBRATION FOR MAC ---
    logical_w, logical_h = pyautogui.size()
    screenshot = pyautogui.screenshot()
    # This magic number handles the Retina display scale (usually 2.0)
    scale = screenshot.width / logical_w
    
    print(f"Mac Scale Factor: {scale}")
    print("--- 5 SECONDS TO SWITCH TO STEINWORLD ---")
    time.sleep(5)

    while True:
        screen = pyautogui.screenshot()
        # Scan every 20th pixel
        for x in range(0, screen.width, 20):
            for y in range(0, screen.height, 20):
                r, g, b = screen.getpixel((x, y))[:3]

                # Adjust these numbers based on your tree color!
                if g > 130 and r < 100 and b < 100:
                    # THE FIX: Divide the pixel coordinate by the scale
                    target_x = x / scale
                    target_y = y / scale
                    
                    print(f"Tree found! Moving to calibrated: {target_x}, {target_y}")
                    
                    # Move smoothly so you can see if it's accurate
                    pyautogui.moveTo(target_x, target_y, duration=0.5)
                    pyautogui.click()
                    
                    # Wait for chopping, then click again for the log
                    time.sleep(6)
                    pyautogui.click() 
                    
                    # Restart the search
                    break
        time.sleep(1)

if __name__ == "__main__":
    try:
        run_calibrated_test()
    except pyautogui.FailSafeException:
        print("\nStopped by user.")