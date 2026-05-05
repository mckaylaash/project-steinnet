import torch
import cv2
import mss
import numpy as np
import pyautogui
from torchvision import transforms
from PIL import Image
import time

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('weights/stein_net_best.pth', weights_only=False) # Path to your saved weights
model.to(device)
model.eval()

# Constants from your proposal
TARGET_SIZE = (224, 224)
ORIG_W, ORIG_H = 1280, 720
LATENCY_THRESHOLD = 0.100 # 100ms

# Same transforms used in training
preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_agent():
    with mss.mss() as sct:
        # Define capture area (match your training screen area)
        monitor = {"top": 0, "left": 0, "width": ORIG_W, "height": ORIG_H}
        
        print("Agent Active. Press Ctrl+C to stop.")
        try:
            while True:
                start_time = time.time()

                # 1. SENSE: Capture screen
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                # 2. THINK: Processing & Inference
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prediction = model(input_tensor) # Outputs normalized (x, y)
                    norm_x, norm_y = prediction[0].cpu().numpy()

                # 3. ACT: Map back to screen pixels and click
                target_x = int(norm_x * ORIG_W)
                target_y = int(norm_y * ORIG_H)
                
                pyautogui.click(target_x, target_y)
                time.sleep(0.1)

                # Check Latency
                loop_time = time.time() - start_time
                if loop_time > LATENCY_THRESHOLD:
                    print(f"Warning: High Latency ({loop_time*1000:.2f}ms)")

        except KeyboardInterrupt:
            print("Agent Stopped.")

run_agent()