# This file is the live inference agent that uses the trained SteinNet model 
# to predict where to click in the game for the chopping task.

import torch
import torch.nn as nn
import mss
import pyautogui
import argparse
from torchvision import models, transforms
from PIL import Image
import time
import threading
import os
from collections import deque
from pynput import keyboard

# load trained model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load("weights/stein_net_best.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()
nav_weights = "weights/path_nav_net_best.pth"

# Inference size from training pipeline
target_size = (224, 224)
latency_threshold = 0.180
start_event = threading.Event()
quit_event = threading.Event()

# keyboard listener for start and stop
def on_press(key):
    try:
        if hasattr(key, "char") and key.char:
            key_char = key.char.lower()
            if key_char == "s":
                start_event.set()
            elif key_char == "q":
                quit_event.set()
    except AttributeError:
        pass

# same transforms used in training
preprocess = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
nav_action_keys = {
    "forward": "w",
    "left": "a",
    "backward": "s",
    "right": "d",
}

# added in function to slightly improve performance, will only assess
# if possibly a tree if it matches these candiates, makes sure 
# agent doenst confuse with water or other variable factors in order to
# improve performance
def is_greenery_candidate(image, x, y):
    w, h = image.size
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    r, g, b = image.getpixel((x, y))[:3]
    water_like = b > g + 20 and b > r + 20
    return (g >= 70) and (g > r + 12) and (g > b + 10) and (not water_like)

# added in function to heuristically estimate path direction for navigation 
# when nav model is unavailable or low confidence.
def estimate_path_action(image, player_x, player_y):
    w, h = image.size
    x_min = max(0, player_x - 220)
    x_max = min(w - 1, player_x + 220)
    y_min = max(0, player_y - 20)
    y_max = min(h - 1, player_y + 120)
    step = 8

    tan_x_values = []
    for y in range(y_min, y_max + 1, step):
        for x in range(x_min, x_max + 1, step):
            r, g, b = image.getpixel((x, y))[:3]
            # Tan/dirt-like path colors.
            is_tan = (
                (r >= 105)
                and (g >= 80)
                and (b >= 45)
                and (r > g > b)
                and ((r - b) >= 18)
            )
            if is_tan:
                tan_x_values.append(x)

    if not tan_x_values:
        return "forward", 0.0

    path_center_x = sum(tan_x_values) / len(tan_x_values)
    offset = path_center_x - player_x
    abs_offset = abs(offset)

    if abs_offset <= 25:
        return "forward", min(1.0, abs_offset / 25.0 + 0.5)
    if offset < 0:
        return "left", min(1.0, abs_offset / 140.0 + 0.35)
    return "right", min(1.0, abs_offset / 140.0 + 0.35)

# loads in nav model weights
def load_nav_model(weights_path, device_):
    payload = torch.load(weights_path, map_location=device_)
    num_classes = int(payload.get("num_classes", 4))
    id_to_action = payload.get(
        "id_to_action",
        {0: "forward", 1: "left", 2: "backward", 3: "right"},
    )
    id_to_action = {int(k): str(v) for k, v in id_to_action.items()}

    nav_model = models.resnet18(weights=None)
    nav_model.fc = nn.Linear(nav_model.fc.in_features, num_classes)
    nav_model.load_state_dict(payload["model_state_dict"])
    nav_model.to(device_)
    nav_model.eval()
    return nav_model, id_to_action

# predicts navigation action using nav model output
def predict_nav_action(nav_model, id_to_action, input_tensor):
    with torch.no_grad():
        logits = nav_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    action = id_to_action.get(int(pred.item()), "forward")
    return action, float(conf.item())

# heuristic function to verify if a pixel candidate is likely to be part of a tree
def is_tree_candidate(image, x, y):
    w, h = image.size
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    trunk_x = x
    trunk_y = min(h - 1, y + 10)
    canopy_x = x
    canopy_y = max(0, y - 35)

    tr, tg, tb = image.getpixel((trunk_x, trunk_y))[:3]
    cr, cg, cb = image.getpixel((canopy_x, canopy_y))[:3]


    trunk_ok = (70 <= tr <= 190) and (40 <= tg <= 140) and (20 <= tb <= 110) and (tr > tg > tb)
    canopy_ok = (cg > cr + 8) and (cg > cb + 8) and (cg >= 70)
    water_like = cb > cg + 25 and cb > cr + 20

    return trunk_ok and canopy_ok and (not water_like)

# searches local area and picks best target pixel 
def pick_best_tree_target(
    image,
    pred_x,
    pred_y,
    player_x,
    player_y,
    search_radius=180,
    step=12,
    prediction_weight=0.6,
    player_weight=1.0,
):
    w, h = image.size
    x_min = max(0, player_x - search_radius)
    x_max = min(w - 1, player_x + search_radius)
    y_min = max(0, player_y - search_radius)
    y_max = min(h - 1, player_y + search_radius)

    best = None
    best_score = float("inf")

    # excludes user interface area from below
    ui_exclusion_y = int(h * 0.90)
    y_max = min(y_max, ui_exclusion_y)

    for y in range(y_min, y_max + 1, step):
        for x in range(x_min, x_max + 1, step):
            if not (is_tree_candidate(image, x, y) or is_greenery_candidate(image, x, y)):
                continue
            d_pred = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5
            d_player = ((x - player_x) ** 2 + (y - player_y) ** 2) ** 0.5
            score = prediction_weight * d_pred + player_weight * d_player
            if score < best_score:
                best_score = score
                best = (x, y)

    return best


def run_agent(
    preview_only=False,
    top=0,
    left=0,
    width=None,
    height=None,
    target_fps=12.0,
    ema_alpha=0.25,
    deadzone_px=12,
    stable_required=3,
    click_cooldown_frames=6,
    tree_check=True,
    debug_actions=False,
    search_radius=180,
    scan_step=12,
    player_x_ratio=0.50,
    player_y_ratio=0.60,
    nav_enabled=True,
    nav_weights=nav_weights,
    nav_action_interval_frames=3,
    nav_conf_threshold=0.45,
    tree_lock_required=4,
    chop_lost_frames=18,
    min_nav_frames_before_chop=36,
):
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01

    nav_model, id_to_action = (None, None)
    if nav_enabled:
        nav_model, id_to_action = load_nav_model(nav_weights, device)

    with mss.MSS() as sct:
        # Capture only the intended game region for lower latency.
        if width is None or height is None:
            primary = sct.monitors[1]
            monitor = {
                "top": primary["top"],
                "left": primary["left"],
                "width": primary["width"],
                "height": primary["height"],
            }
        else:
            monitor = {"top": top, "left": left, "width": width, "height": height}
        orig_w, orig_h = monitor["width"], monitor["height"]

        # pyautogui clicks use logical coordinates on macOS (Retina aware).
        logical_w, logical_h = pyautogui.size()
        scale_x = orig_w / logical_w
        scale_y = orig_h / logical_h

        mode_text = "PREVIEW (move only, no clicks)" if preview_only else "LIVE (click enabled)"
        frame_budget = 1.0 / max(target_fps, 1.0)
        latency_history = deque(maxlen=120)
        # stabilization state
        smooth_x = None
        smooth_y = None
        stable_frames = 0
        last_click_frame = -10_000

        print("Agent Active.")
        print("Press 'S' to START the agent.")
        print("Press 'Q' to QUIT at any time.")

        start_event.clear()
        quit_event.clear()

        with keyboard.Listener(on_press=on_press):
            # key press start
            while not start_event.is_set() and not quit_event.is_set():
                time.sleep(0.05)
            if quit_event.is_set():
                print("User requested quit before start.")
                return
            print("Starting Loop")

            print(f"Mode: {mode_text}")
            print(f"Device: {device}, Capture: {orig_w}x{orig_h} at ({left}, {top}), Logical: {logical_w}x{logical_h}")
            print(f"Target FPS: {target_fps:.1f} (frame budget: {frame_budget*1000:.1f} ms)")
            print(f"Behavior: {'NAV+CHOP' if nav_enabled else 'CHOP-only'}")

            try:
                frame_idx = 0
                state = "NAVIGATE" if nav_enabled else "CHOP"
                nav_frames = 0
                tree_lock_frames = 0
                tree_miss_frames = 0
                while True:
                    # failsafe q key
                    if quit_event.is_set():
                        print("User requested quit.")
                        break

                    frame_idx += 1
                    start_time = time.perf_counter()

                    # capture screen
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                    # preprocess captured image
                    input_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    # predict and normalize coord to click
                    with torch.no_grad():
                        prediction = model(input_tensor) # Outputs normalized (x, y)
                        norm_x, norm_y = prediction[0].cpu().numpy()
                        norm_x = float(min(max(norm_x, 0.0), 1.0))
                        norm_y = float(min(max(norm_y, 0.0), 1.0))

                    # stabilize coords
                    if smooth_x is None:
                        smooth_x, smooth_y = norm_x, norm_y
                    else:
                        smooth_x = ema_alpha * norm_x + (1.0 - ema_alpha) * smooth_x
                        smooth_y = ema_alpha * norm_y + (1.0 - ema_alpha) * smooth_y

                    raw_px_x = int((norm_x * orig_w) / scale_x)
                    raw_px_y = int((norm_y * orig_h) / scale_y)
                    smooth_px_x = int((smooth_x * orig_w) / scale_x)
                    smooth_px_y = int((smooth_y * orig_h) / scale_y)
                    player_px_x = int(player_x_ratio * orig_w)
                    player_px_y = int(player_y_ratio * orig_h)

                    raw_step = abs(raw_px_x - smooth_px_x) + abs(raw_px_y - smooth_px_y)
                    if raw_step <= deadzone_px:
                        stable_frames += 1
                    else:
                        stable_frames = 0
                    stable_ok = stable_frames >= stable_required

                    target_x, target_y = smooth_px_x, smooth_px_y
                    tree_ok = False
                    if tree_check:
                        focused_tree = pick_best_tree_target(
                            img,
                            pred_x=int(smooth_x * orig_w),
                            pred_y=int(smooth_y * orig_h),
                            player_x=player_px_x,
                            player_y=player_px_y,
                            search_radius=search_radius,
                            step=scan_step,
                        )
                        if focused_tree is not None:
                            # Map selected physical-pixel target back to logical screen coords.
                            target_x = int(focused_tree[0] / scale_x)
                            target_y = int(focused_tree[1] / scale_y)
                            tree_ok = True
                        else:
                            tree_ok = False
                    else:
                        tree_ok = True

                    if state == "NAVIGATE":
                        nav_frames += 1
                        if tree_ok and stable_ok:
                            tree_lock_frames += 1
                        else:
                            tree_lock_frames = 0
                        
                        # makes sure it doesn't chop immedietly and actually navigates 
                        if nav_frames >= min_nav_frames_before_chop and tree_lock_frames >= tree_lock_required:
                            state = "CHOP"
                            nav_frames = 0
                            tree_miss_frames = 0
                            if debug_actions:
                                print("[state] NAVIGATE -> CHOP")
                        elif nav_enabled and frame_idx % max(nav_action_interval_frames, 1) == 0:
                            path_action, path_conf = estimate_path_action(img, player_px_x, player_px_y)
                            if nav_model is None:
                                if not preview_only:
                                    pyautogui.press(path_action if path_action in nav_action_keys else "w")
                                if debug_actions:
                                    key_used = nav_action_keys.get(path_action, "w")
                                    print(f"[nav] fallback action={path_action} key={key_used}")
                                continue
                            nav_action, nav_conf = predict_nav_action(nav_model, id_to_action, input_tensor)

                            chosen_action = nav_action
                            chosen_conf = nav_conf
                            if path_conf >= 0.55:
                                chosen_action = path_action
                                chosen_conf = path_conf

                            if chosen_conf >= nav_conf_threshold:
                                key_to_press = nav_action_keys.get(chosen_action)
                                if key_to_press:
                                    if not preview_only:
                                        pyautogui.press(key_to_press)
                                    if debug_actions:
                                        print(
                                            f"[nav] action={chosen_action} conf={chosen_conf:.2f} "
                                            f"(model={nav_action}:{nav_conf:.2f}, path={path_action}:{path_conf:.2f}) key={key_to_press}"
                                        )
                            elif debug_actions and frame_idx % 10 == 0:
                                print(
                                    f"[nav] low confidence: chosen={chosen_action}:{chosen_conf:.2f} "
                                    f"(model={nav_action}:{nav_conf:.2f}, path={path_action}:{path_conf:.2f})"
                                )

                    if state == "CHOP":
                        if not tree_ok:
                            tree_miss_frames += 1
                            if nav_enabled and tree_miss_frames >= chop_lost_frames:
                                state = "NAVIGATE"
                                nav_frames = 0
                                tree_lock_frames = 0
                                if debug_actions:
                                    print("[state] CHOP -> NAVIGATE")
                        else:
                            tree_miss_frames = 0

                            cooldown_ok = (frame_idx - last_click_frame) >= click_cooldown_frames
                            ready_to_click = stable_ok and cooldown_ok
                            if ready_to_click and tree_ok:
                                pyautogui.click(target_x, target_y)
                                last_click_frame = frame_idx
                                if debug_actions:
                                    print(
                                        f"[action] click target=({target_x}, {target_y}) "
                                        f"stable={stable_frames} raw_step={raw_step}"
                                    )
                            elif debug_actions and frame_idx % 10 == 0:
                                print(
                                    f"[action] skip stable_ok={stable_ok} cooldown_ok={cooldown_ok} "
                                    f"tree_ok={tree_ok} stable={stable_frames} raw_step={raw_step}"
                                )

                    # latency check so computer doesn't overheat
                    loop_time = time.perf_counter() - start_time
                    latency_history.append(loop_time * 1000.0)
                    if frame_idx % 30 == 0 and latency_history:
                        sorted_lat = sorted(latency_history)
                        p95_idx = int(0.95 * (len(sorted_lat) - 1))
                        p95_ms = sorted_lat[p95_idx]
                        avg_ms = sum(latency_history) / len(latency_history)
                        warn = " [HIGH]" if p95_ms > (latency_threshold * 1000.0) else ""
                        print(f"[latency] avg={avg_ms:.1f}ms p95={p95_ms:.1f}ms{warn}")

                    sleep_needed = frame_budget - loop_time
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)

            except KeyboardInterrupt:
                print("Agent Stopped.")

if __name__ == "__main__":
    run_agent(
        preview_only=False,
        top=0,
        left=0,
        width=None,
        height=None,
        target_fps=12.0,
        ema_alpha=0.25,
        deadzone_px=12,
        stable_required=3,
        click_cooldown_frames=6,
        tree_check=True,
        debug_actions=False,
        search_radius=180,
        scan_step=12,
        player_x_ratio=0.50,
        player_y_ratio=0.60,
        nav_enabled=True,
        nav_weights=nav_weights,
        nav_action_interval_frames=3,
        nav_conf_threshold=0.45,
        tree_lock_required=4,
        chop_lost_frames=18,
        min_nav_frames_before_chop=36,
    )