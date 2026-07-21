# infer.py
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from sms import send_accident_alert  # <--- import our SMS function


def get_device(device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device):
    num_classes = 2  # ['negative', 'positive']
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def infer_video(model,
                device,
                video_path: str,
                frame_stride: int = 5,
                threshold: float = 0.5):
    """
    Returns:
        final_label (str): 'positive' or 'negative'
        mean_probs (np.array): [p_negative, p_positive]
        first_accident_time_str (str or None): time in seconds where model first saw an accident
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] Using frame_stride={frame_stride}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    transform = get_transform()

    probs_list = []
    first_accident_time = None

    frame_idx = 0
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every Nth frame
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = transform(frame_rgb).unsqueeze(0).to(device)

            logits = model(x)
            probs = softmax(logits).cpu().numpy()[0]  # [p_neg, p_pos]
            probs_list.append(probs)

            # If this frame is strongly positive and we didn't log a time yet
            if probs[1] >= threshold and first_accident_time is None:
                # frame index to seconds
                seconds = frame_idx / fps if fps > 0 else 0
                first_accident_time = seconds

            frame_idx += 1

    cap.release()

    if len(probs_list) == 0:
        raise RuntimeError("No frames were processed. Check your video or frame_stride.")

    probs_arr = np.stack(probs_list, axis=0)
    mean_probs = probs_arr.mean(axis=0)  # average over frames
    p_negative, p_positive = mean_probs

    final_label = "positive" if p_positive >= threshold else "negative"

    print("\n===== VIDEO RESULT =====")
    print(f"Raw predicted class: {final_label} (conf={p_positive if final_label=='positive' else p_negative:.4f})")
    print(f"Mean probs -> negative: {p_negative:.4f}, positive: {p_positive:.4f}")
    print(f"Final label (using threshold={threshold:.2f} on positive prob): {final_label.upper()}")
    if first_accident_time is not None:
        print(f"First accident frame (>= threshold) at ~{first_accident_time:.2f} seconds")
    print("========================\n")

    # Prepare first_accident_time as string for SMS (optional)
    first_accident_time_str = (
        f"{first_accident_time:.2f} s"
        if first_accident_time is not None
        else None
    )

    return final_label, mean_probs, first_accident_time_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video file"
    )
    parser.add_argument(
        "--model-path",
        default="models/accident_frame_resnet18.pth",
        help="Path to the trained model .pth"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Use 1 out of every N frames"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on positive probability"
    )
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model_path, device)

    final_label, mean_probs, first_acc_time_str = infer_video(
        model=model,
        device=device,
        video_path=args.video,
        frame_stride=args.frame_stride,
        threshold=args.threshold,
    )

    # --------------- ALERT TRIGGER ---------------
    # If accident detected with decent confidence, send SMS
    p_positive = float(mean_probs[1])

    if final_label == "positive" and p_positive >= args.threshold:
        print("[ALERT] Accident detected. Triggering SMS alerts...")
        send_accident_alert(
            video_path=args.video,
            positive_prob=p_positive,
            first_accident_time=first_acc_time_str,
        )
    else:
        print("[INFO] No accident detected above threshold. No SMS sent.")


if __name__ == "__main__":
    main()
