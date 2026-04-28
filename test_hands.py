import pyautogui
import time

# 1. THE PANIC BUTTON
# If it goes off-track, slam mouse to the TOP-LEFT corner!
pyautogui.FAILSAFE = True

def run_steinnet_v2():
    # --- MAC CALIBRATION ---
    # Standard for MacBook Air is 2.0 (Physical pixels vs Logical mouse pixels)
    logical_w, logical_h = pyautogui.size()
    screenshot = pyautogui.screenshot()
    scale = screenshot.width / logical_w
    
    print(f"Detected Scale: {scale}")
    print("Starting in 7 seconds... Stand near some trees!")
    time.sleep(7)

    while True:
        # Take a fresh screenshot
        screen = pyautogui.screenshot()
        w, h = screen.size
        
        # 2. SEARCH ZONE & PROXIMITY LOGIC
        # We define a box around the center (where your character is)
        zone_size = 350 
        center_x, center_y = w // 2, h // 2
        
        # We scan from the BOTTOM of the box UP to the CENTER.
        # This ensures we click the tree closest to our feet first.
        found = False
        
        # Scan Y from (bottom) to (middle)
        for y in range(center_y + zone_size, center_y - 100, -15): 
            # Scan X from left to right
            for x in range(center_x - zone_size, center_x + zone_size, 15):
                
                # Stay within screen bounds
                if x >= w or y >= h or x < 0 or y < 0: continue
                
                r, g, b = screen.getpixel((x, y))[:3]

                # 3. TRUNK COLOR DETECTION (Browns/Greys)
                # Trunks are usually balanced (R, G, and B are similar) 
                # and darker than the bright grass.
                is_neutral = abs(r - g) < 20 and abs(g - b) < 20
                is_dark_enough = 50 < g < 130
                
                if is_neutral and is_dark_enough:
                    # CALIBRATE FOR MAC
                    target_x = x / scale
                    target_y = y / scale
                    
                    print(f"Targeting tree base at: {target_x}, {target_y}")
                    pyautogui.moveTo(target_x, target_y, duration=0.2)
                    pyautogui.click()
                    
                    # 4. ANIMATION WAIT & LOOT
                    # Wait 6 seconds for the chopping to finish
                    print("Chopping... waiting for log drop.")
                    time.sleep(6)
                    
                    # Click again to pick up the log
                    pyautogui.click() 
                    
                    found = True
                    break
            if found: break
        
        # If no trees are nearby, wait a second before scanning again
        if not found:
            print("No reachable trees found. Searching...")
            time.sleep(2)

if __name__ == "__main__":
    try:
        run_steinnet_v2()
    except pyautogui.FailSafeException:
        print("\nFail-safe triggered. Script stopped.")