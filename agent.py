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

# Load your trained model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load("weights/stein_net_best.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()
NAV_WEIGHTS_DEFAULT = "weights/path_nav_net_best.pth"

# Inference size from training pipeline
TARGET_SIZE = (224, 224)
LATENCY_THRESHOLD = 0.180
start_event = threading.Event()
quit_event = threading.Event()


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

# Same transforms used in training
preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
NAV_ACTION_KEYS = {
    "forward": "w",
    "left": "a",
    "backward": "s",
    "right": "d",
}


def load_nav_model(weights_path, device_):
    if not weights_path or not os.path.exists(weights_path):
        return None, None
    payload = torch.load(weights_path, map_location=device_)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        return None, None
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


def predict_nav_action(nav_model, id_to_action, input_tensor):
    with torch.no_grad():
        logits = nav_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    action = id_to_action.get(int(pred.item()), "forward")
    return action, float(conf.item())


def is_tree_candidate(image, x, y):
    """Heuristic gate to reduce shrub/rock/water misclicks."""
    w, h = image.size
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    trunk_x = x
    trunk_y = min(h - 1, y + 10)
    canopy_x = x
    canopy_y = max(0, y - 35)

    tr, tg, tb = image.getpixel((trunk_x, trunk_y))[:3]
    cr, cg, cb = image.getpixel((canopy_x, canopy_y))[:3]

    # Trunk tends to be warm brown and darker than foliage.
    trunk_ok = (70 <= tr <= 190) and (40 <= tg <= 140) and (20 <= tb <= 110) and (tr > tg > tb)
    # Leaves above trunk are usually green-dominant in this biome.
    canopy_ok = (cg > cr + 8) and (cg > cb + 8) and (cg >= 70)
    # Reject obvious water-like blue patches.
    water_like = cb > cg + 25 and cb > cr + 20

    return trunk_ok and canopy_ok and (not water_like)


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
    """Pick a tree-like target near player and near model prediction."""
    w, h = image.size
    x_min = max(0, player_x - search_radius)
    x_max = min(w - 1, player_x + search_radius)
    y_min = max(0, player_y - search_radius)
    y_max = min(h - 1, player_y + search_radius)

    best = None
    best_score = float("inf")

    # Exclude chat/hotbar UI region near the bottom edge.
    ui_exclusion_y = int(h * 0.90)
    y_max = min(y_max, ui_exclusion_y)

    for y in range(y_min, y_max + 1, step):
        for x in range(x_min, x_max + 1, step):
            if not is_tree_candidate(image, x, y):
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
    nav_weights=NAV_WEIGHTS_DEFAULT,
    nav_action_interval_frames=3,
    nav_conf_threshold=0.45,
    tree_lock_required=4,
    chop_lost_frames=18,
):
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01

    nav_model, id_to_action = (None, None)
    if nav_enabled:
        nav_model, id_to_action = load_nav_model(nav_weights, device)
        if nav_model is None:
            print(f"[warn] Nav model not loaded from: {nav_weights}. Running CHOP-only mode.")
            nav_enabled = False

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
        # Stabilization state
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
            # KEY PRESS START: Wait for user to hit 's'
            while not start_event.is_set() and not quit_event.is_set():
                time.sleep(0.05)
            if quit_event.is_set():
                print("User requested quit before start.")
                return
            print("Starting Loop...")

            print(f"Mode: {mode_text}")
            print(f"Device: {device}, Capture: {orig_w}x{orig_h} at ({left}, {top}), Logical: {logical_w}x{logical_h}")
            print(f"Target FPS: {target_fps:.1f} (frame budget: {frame_budget*1000:.1f} ms)")
            print(f"Behavior: {'NAV+CHOP' if nav_enabled else 'CHOP-only'}")

            try:
                frame_idx = 0
                state = "NAVIGATE" if nav_enabled else "CHOP"
                tree_lock_frames = 0
                tree_miss_frames = 0
                while True:
                    # FAILSAFE Q PRESS: Check if 'q' is pressed
                    if quit_event.is_set():
                        print("User requested quit.")
                        break

                    frame_idx += 1
                    start_time = time.perf_counter()

                    # 1. SENSE: Capture screen
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                    # 2. THINK: Processing & Inference
                    input_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        prediction = model(input_tensor) # Outputs normalized (x, y)
                        norm_x, norm_y = prediction[0].cpu().numpy()
                        norm_x = float(min(max(norm_x, 0.0), 1.0))
                        norm_y = float(min(max(norm_y, 0.0), 1.0))

                    # 3. STABILIZE: smooth and apply deadzone before acting.
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

                    target_x, target_y = smooth_px_x, smooth_px_y
                    tree_ok = True
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

                    if state == "NAVIGATE":
                        if tree_ok:
                            tree_lock_frames += 1
                        else:
                            tree_lock_frames = 0

                        if tree_lock_frames >= tree_lock_required:
                            state = "CHOP"
                            tree_miss_frames = 0
                            if debug_actions:
                                print("[state] NAVIGATE -> CHOP")
                        elif nav_enabled and frame_idx % max(nav_action_interval_frames, 1) == 0:
                            nav_action, nav_conf = predict_nav_action(nav_model, id_to_action, input_tensor)
                            if nav_conf >= nav_conf_threshold:
                                key_to_press = NAV_ACTION_KEYS.get(nav_action)
                                if key_to_press:
                                    if not preview_only:
                                        pyautogui.press(key_to_press)
                                    if debug_actions:
                                        print(f"[nav] action={nav_action} conf={nav_conf:.2f} key={key_to_press}")
                            elif debug_actions and frame_idx % 10 == 0:
                                print(f"[nav] low confidence: action={nav_action} conf={nav_conf:.2f}")

                    if state == "CHOP":
                        if not tree_ok:
                            tree_miss_frames += 1
                            if nav_enabled and tree_miss_frames >= chop_lost_frames:
                                state = "NAVIGATE"
                                tree_lock_frames = 0
                                if debug_actions:
                                    print("[state] CHOP -> NAVIGATE")
                        else:
                            tree_miss_frames = 0

                        if preview_only:
                            # Move cursor only so you can validate aim safely.
                            pyautogui.moveTo(target_x, target_y, duration=0)
                            if frame_idx % 10 == 0:
                                print(
                                    f"[preview] state={state} target=({target_x}, {target_y}) "
                                    f"raw=({raw_px_x}, {raw_px_y}) stable={stable_frames} "
                                    f"tree_ok={tree_ok} player=({player_px_x},{player_px_y})"
                                )
                        else:
                            stable_ok = stable_frames >= stable_required
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

                    # Check Latency
                    loop_time = time.perf_counter() - start_time
                    latency_history.append(loop_time * 1000.0)
                    if frame_idx % 30 == 0 and latency_history:
                        sorted_lat = sorted(latency_history)
                        p95_idx = int(0.95 * (len(sorted_lat) - 1))
                        p95_ms = sorted_lat[p95_idx]
                        avg_ms = sum(latency_history) / len(latency_history)
                        warn = " [HIGH]" if p95_ms > (LATENCY_THRESHOLD * 1000.0) else ""
                        print(f"[latency] avg={avg_ms:.1f}ms p95={p95_ms:.1f}ms{warn}")

                    # Frame pacing to keep behavior stable and avoid maxing CPU/GPU.
                    sleep_needed = frame_budget - loop_time
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)

            except KeyboardInterrupt:
                print("Agent Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SteinNet live inference agent.")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Move cursor to predicted target without clicking.",
    )
    parser.add_argument("--top", type=int, default=0, help="Capture region top coordinate.")
    parser.add_argument("--left", type=int, default=0, help="Capture region left coordinate.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Capture region width. Omit to use full primary display.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Capture region height. Omit to use full primary display.",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Target control-loop FPS.")
    parser.add_argument("--ema-alpha", type=float, default=0.25, help="EMA smoothing factor (0-1).")
    parser.add_argument("--deadzone-px", type=int, default=12, help="Ignore tiny prediction jitter under this pixel delta.")
    parser.add_argument("--stable-frames", type=int, default=3, help="Frames required in deadzone before clicking.")
    parser.add_argument("--click-cooldown-frames", type=int, default=6, help="Minimum frames between clicks.")
    parser.add_argument("--no-tree-check", action="store_true", help="Disable tree-color verification gate.")
    parser.add_argument("--debug-actions", action="store_true", help="Print why clicks fire or are skipped.")
    parser.add_argument("--search-radius", type=int, default=180, help="Local search radius around player (pixels).")
    parser.add_argument("--scan-step", type=int, default=12, help="Pixel step size for tree candidate scan.")
    parser.add_argument("--player-x-ratio", type=float, default=0.50, help="Player X anchor as capture-width ratio.")
    parser.add_argument("--player-y-ratio", type=float, default=0.60, help="Player Y anchor as capture-height ratio.")
    args = parser.parse_args()
    run_agent(
        preview_only=args.preview_only,
        top=args.top,
        left=args.left,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        ema_alpha=args.ema_alpha,
        deadzone_px=args.deadzone_px,
        stable_required=args.stable_frames,
        click_cooldown_frames=args.click_cooldown_frames,
        tree_check=not args.no_tree_check,
        debug_actions=args.debug_actions,
        search_radius=args.search_radius,
        scan_step=args.scan_step,
        player_x_ratio=args.player_x_ratio,
        player_y_ratio=args.player_y_ratio,
    )