import pyautogui
import time

# Built-in fail-safe: Slam mouse to TOP-LEFT corner to kill script
pyautogui.FAILSAFE = True

def run_simple_test():
    # Use 1.0 if you changed your Mac display settings, or 2.0 for Retina
    scale = 2.0 
    
    print("--- STARTING SEARCH ---")
    print("EMERGENCY STOP: Slam mouse to TOP-LEFT corner!")
    time.sleep(5)

    while True:
        # 1. Take a small screenshot of just the CENTER of the screen
        # This is faster and avoids clicking browser tabs/logout buttons
        screen = pyautogui.screenshot()
        w, h = screen.size
        
        # Define a 'Search Box' in the middle
        search_area = 400 
        left = int((w/2) - search_area)
        top = int((h/2) - search_area)

        # 2. Scan the box
        found = False
        for x in range(left, left + (search_area * 2), 20):
            for y in range(top, top + (search_area * 2), 20):
                r, g, b = screen.getpixel((x, y))[:3]

                # RELAXED GREEN: Is Green the strongest color?
                if g > r and g > b and g > 100:
                    print(f"Green found at {x}, {y}")
                    pyautogui.moveTo(x/scale, y/scale, duration=0.2)
                    pyautogui.click()
                    time.sleep(6) # Chop time
                    pyautogui.click() # Loot time
                    found = True
                    break
            if found: break