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

    # 1. DEFINE THE 'INTERACTION ZONE'
    # We only care about pixels near the center of the screen
    center_x, center_y = (logical_w * scale) / 2, (logical_h * scale) / 2
    zone_size = 300 # This is a box around your character
    
    start_x = int(center_x - zone_size)
    end_x = int(center_x + zone_size)
    start_y = int(center_y - zone_size)
    end_y = int(center_y + zone_size)

    while True:
        screen = pyautogui.screenshot()
        # Scan every 15th pixel
        for x in range(start_x, end_x, 15):
            for y in range(start_y, end_y, 15):
                r, g, b = screen.getpixel((x, y))[:3]

                # Adjust these numbers based on your tree color!
                if g > 160 and r < 70 and b < 70:
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