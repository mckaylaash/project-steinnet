import pyautogui
import time

pyautogui.FAILSAFE = True

def run_steinnet_v3():
    logical_w, logical_h = pyautogui.size()
    screenshot = pyautogui.screenshot()
    scale = screenshot.width / logical_w
    
    # COLORS
    TARGET_BROWN = (137, 120, 87)
    TOLERANCE = 25

    print(f"Calibrated Scale: {scale}")
    print("Starting... Switch to Steinworld!")
    time.sleep(7)

    while True:
        screen = pyautogui.screenshot()
        w, h = screen.size
        
        center_x, center_y = w // 2, h // 2
        zone_size = 350 
        found = False

        # Scan BOTTOM-UP
        for y in range(center_y + zone_size, center_y - 100, -15): 
            for x in range(center_x - zone_size, center_x + zone_size, 15):
                
                if x >= w or y >= h or y < 60: continue
                r, g, b = screen.getpixel((x, y))[:3]

                # 1. TRUNK CHECK (The Brown)
                is_brown = (abs(r - TARGET_BROWN[0]) < TOLERANCE and 
                            abs(g - TARGET_BROWN[1]) < TOLERANCE and 
                            abs(b - TARGET_BROWN[2]) < TOLERANCE)
                
                # 2. GREEN CHECK (The Leaves Above)
                # We check a pixel slightly above the current point to verify it's a tree
                r_above, g_above, b_above = screen.getpixel((x, y - 50))[:3]
                has_leaves_above = g_above > r_above and g_above > 100

                # 3. ROCK REJECTION
                is_not_rock = b < 110 

                if is_brown and has_leaves_above and is_not_rock:
                    target_x = x / scale
                    target_y = y / scale
                    
                    print(f"Tree verified (Trunk + Leaves)! Clicking: {target_x}, {target_y}")
                    pyautogui.moveTo(target_x, target_y, duration=0.2)
                    pyautogui.click()
                    
                    time.sleep(6) # Chop
                    pyautogui.click() # Loot
                    found = True
                    break
            if found: break
        
        if not found:
            time.sleep(1)

if __name__ == "__main__":
    try:
        run_steinnet_v3()
    except pyautogui.FailSafeException:
        print("\nFail-safe triggered.")