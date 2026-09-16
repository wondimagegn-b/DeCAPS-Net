"""Extract colorized body-part parsing maps from GFBMD videos."""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

NUM_FRAMES_PER_VIDEO = 8
PARSE_CROP_MARGIN_RATIO = 3.0
PERSON_CLASS_ID = 0
YOLO_CONF = 0.25
SAPIENS_INPUT_HW = (1024, 768)

ORIGINAL_GOLIATH_CLASSES = (
    "Background", "Apparel", "Chair", "Eyeglass_Frame", "Eyeglass_Lenses",
    "Face_Neck", "Hair", "Headset", "Left_Foot", "Left_Hand",
    "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock",
    "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing", "Lower_Spandex",
    "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg",
    "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg",
    "Torso", "Upper_Clothing", "Visible_Badge", "Lower_Lip", "Upper_Lip",
    "Lower_Teeth", "Upper_Teeth", "Tongue",
)

ORIGINAL_GOLIATH_PALETTE = [
    [50, 50, 50], [255, 218, 0], [102, 204, 0], [14, 0, 204],
    [0, 204, 160], [128, 200, 255], [255, 0, 109], [0, 255, 36],
    [189, 0, 204], [255, 0, 218], [0, 160, 204], [0, 255, 145],
    [204, 0, 131], [182, 0, 255], [255, 109, 0], [0, 255, 255],
    [72, 0, 255], [204, 43, 0], [204, 131, 0], [255, 0, 0],
    [72, 255, 0], [189, 204, 0], [182, 255, 0], [102, 0, 204],
    [32, 72, 204], [0, 145, 255], [14, 204, 0], [0, 128, 72],
    [204, 0, 43], [235, 205, 119], [115, 227, 112], [157, 113, 143],
    [132, 93, 50], [82, 21, 114],
]

REMOVE_CLASSES = (
    "Eyeglass_Frame", "Eyeglass_Lenses", "Visible_Badge", "Chair",
    "Lower_Spandex", "Headset",
)

GOLIATH_PALETTE = [
    ORIGINAL_GOLIATH_PALETTE[i]
    for i, name in enumerate(ORIGINAL_GOLIATH_CLASSES)
    if name not in REMOVE_CLASSES
]
PALETTE_ARRAY = np.array(GOLIATH_PALETTE, dtype=np.uint8)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def natural_subject_sort_key(name: str):
    s = str(name).strip()
    return (0, int(s)) if s.isdigit() else (1, s.lower())


def list_videos(dataset_root: str) -> List[Dict]:
    entries = []
    groups = [
        ("ASD", 1, os.path.join(dataset_root, "Autism", "children with ASD")),
        ("TD", 0, os.path.join(dataset_root, "Typical")),
    ]
    for label_name, label_id, root in groups:
        if not os.path.isdir(root):
            print(f"[WARN] Missing folder: {root}")
            continue
        subjects = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
        for subject_id in sorted(subjects, key=natural_subject_sort_key):
            video_path = os.path.join(root, subject_id, "video", "video.avi")
            if os.path.isfile(video_path):
                entries.append({
                    "label_name": label_name,
                    "label_id": label_id,
                    "subject_id": str(subject_id),
                    "video_path": video_path,
                })
            else:
                print(f"[WARN] Video not found: {video_path}")
    return entries


def sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if num_frames <= 1:
        return [0]
    indices = np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int).tolist()
    return [min(max(i, 0), total_frames - 1) for i in indices]


def read_frames_by_indices(video_path: str, indices: List[int]) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((idx, frame))
    cap.release()
    return frames


def rescale_bbox(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int,
    margin_ratio: float = PARSE_CROP_MARGIN_RATIO,
) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(round(bw * margin_ratio))
    pad_h = int(round(bh * margin_ratio))
    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(w - 1, x2 + pad_w),
        min(h - 1, y2 + pad_h),
    )


def pick_largest_person_bbox(yolo_result, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int, float]]:
    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_area = -1.0
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        if clss[i] != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = xyxy[i]
        x1 = int(max(0, min(img_w - 1, round(x1))))
        y1 = int(max(0, min(img_h - 1, round(y1))))
        x2 = int(max(0, min(img_w - 1, round(x2))))
        y2 = int(max(0, min(img_h - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2, float(confs[i]))
    return best


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    label_map = np.clip(label_map.astype(np.int64), 0, len(PALETTE_ARRAY) - 1)
    return PALETTE_ARRAY[label_map]


class SapiensSegmenter:
    def __init__(self, checkpoint_path: str, device: str):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Sapiens checkpoint not found: {checkpoint_path}")
        self.device = device
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(SAPIENS_INPUT_HW),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.5 / 255, 116.5 / 255, 103.5 / 255],
                std=[58.5 / 255, 57.0 / 255, 57.5 / 255],
            ),
        ])

    @torch.inference_mode()
    def predict_label_map(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        out = self.model(inp)
        out = F.interpolate(
            out,
            size=(crop_rgb.shape[0], crop_rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        return torch.argmax(out, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def crop_after_parsing(
    label_map: np.ndarray,
    crop_bgr: np.ndarray,
    padding_ratio: float = 0.08,
    min_fg_pixels: int = 300,
    min_box_hw: Tuple[int, int] = (40, 20),
    min_height_ratio: float = 0.55,
    min_width_ratio: float = 0.20,
    max_trim_top: float = 0.12,
    max_trim_bottom: float = 0.08,
    max_trim_left: float = 0.18,
    max_trim_right: float = 0.18,
):
    if label_map is None or label_map.size == 0 or crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, label_map, False

    Hc, Wc = label_map.shape[:2]
    fg_mask = (label_map != 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    if int(fg_mask.sum()) < min_fg_pixels:
        return crop_bgr, label_map, False

    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return crop_bgr, label_map, False

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    if bh < min_box_hw[0] or bw < min_box_hw[1]:
        return crop_bgr, label_map, False
    if bh / max(1, Hc) < min_height_ratio or bw / max(1, Wc) < min_width_ratio:
        return crop_bgr, label_map, False

    pad_x = max(2, int(round(bw * padding_ratio)))
    pad_y = max(2, int(round(bh * padding_ratio)))
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(Wc - 1, x2 + pad_x)
    py2 = min(Hc - 1, y2 + pad_y)

    left_limit = int(round(Wc * max_trim_left))
    right_limit = Wc - 1 - int(round(Wc * max_trim_right))
    top_limit = int(round(Hc * max_trim_top))
    bottom_limit = Hc - 1 - int(round(Hc * max_trim_bottom))

    fx1 = max(0, min(min(px1, left_limit), Wc - 1))
    fy1 = max(0, min(min(py1, top_limit), Hc - 1))
    fx2 = max(0, min(max(px2, right_limit), Wc - 1))
    fy2 = max(0, min(max(py2, bottom_limit), Hc - 1))

    if fx2 <= fx1 or fy2 <= fy1:
        return crop_bgr, label_map, False

    refined_crop = crop_bgr[fy1:fy2 + 1, fx1:fx2 + 1].copy()
    refined_label = label_map[fy1:fy2 + 1, fx1:fx2 + 1].copy()
    if refined_crop.size == 0 or refined_label.size == 0:
        return crop_bgr, label_map, False
    return refined_crop, refined_label, True


def process_dataset(args) -> None:
    ensure_dir(args.output_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.yolo_weights)
    parser = SapiensSegmenter(args.sapiens_ckpt, device)
    entries = list_videos(args.dataset_root)
    if not entries:
        print("No videos found.")
        return

    print(f"Found {len(entries)} videos.")
    detector_device = 0 if device == "cuda" else device

    for item in tqdm(entries, desc="Videos"):
        label_name = item["label_name"]
        subject_id = item["subject_id"]
        video_path = item["video_path"]
        out_subj_root = os.path.join(args.output_root, label_name, subject_id, "parsing")
        color_dir = os.path.join(out_subj_root, "colorized")
        ensure_dir(color_dir)

        cap_info = cv2.VideoCapture(video_path)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap_info.get(cv2.CAP_PROP_FPS))
        cap_info.release()

        frame_indices = sample_frame_indices(total_frames, args.num_frames)
        frames = read_frames_by_indices(video_path, frame_indices)
        if not frames:
            print(f"[WARN] No readable sampled frames: {video_path}")
            continue

        subject_meta = {
            "label_name": label_name,
            "label_id": item["label_id"],
            "subject_id": subject_id,
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "sampled_indices": frame_indices,
            "saved_frames": [],
        }

        for order_i, (frame_idx, frame_bgr) in enumerate(frames):
            H, W = frame_bgr.shape[:2]
            results = detector.predict(
                source=frame_bgr,
                conf=YOLO_CONF,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                device=detector_device,
            )
            picked = pick_largest_person_bbox(results[0], W, H) if results else None

            if picked is None:
                px1, py1, px2, py2 = 0, 0, W - 1, H - 1
                raw_x1, raw_y1, raw_x2, raw_y2 = 0, 0, W - 1, H - 1
                conf = 0.0
            else:
                raw_x1, raw_y1, raw_x2, raw_y2, conf = picked
                px1, py1, px2, py2 = rescale_bbox(raw_x1, raw_y1, raw_x2, raw_y2, W, H)

            parse_crop = frame_bgr[py1:py2 + 1, px1:px2 + 1].copy()
            if parse_crop.size == 0:
                continue
            label_map_full = parser.predict_label_map(parse_crop)

            rx1 = max(0, raw_x1 - px1)
            ry1 = max(0, raw_y1 - py1)
            rx2 = min(parse_crop.shape[1] - 1, raw_x2 - px1)
            ry2 = min(parse_crop.shape[0] - 1, raw_y2 - py1)
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            crop = parse_crop[ry1:ry2 + 1, rx1:rx2 + 1].copy()
            label_map = label_map_full[ry1:ry2 + 1, rx1:rx2 + 1].copy()
            if crop.size == 0 or label_map.size == 0:
                continue

            refined_crop, refined_label_map, parsing_ok = crop_after_parsing(label_map, crop)
            if refined_crop is None or refined_label_map is None or refined_crop.size == 0 or refined_label_map.size == 0:
                continue
            if parsing_ok:
                crop = refined_crop
                label_map = refined_label_map

            colorized_rgb = colorize_label_map(label_map)
            if colorized_rgb.size == 0:
                continue

            base = f"f{order_i:03d}_idx{frame_idx:05d}"
            color_path = os.path.join(color_dir, base + "_color.png")
            cv2.imwrite(color_path, cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR))
            subject_meta["saved_frames"].append({
                "order_index": order_i,
                "video_frame_index": int(frame_idx),
                "bbox_xyxy": [int(raw_x1), int(raw_y1), int(raw_x2), int(raw_y2)],
                "det_conf": float(conf),
                "crop_size_hw": [int(crop.shape[0]), int(crop.shape[1])],
                "colorized_path": color_path,
            })

        with open(os.path.join(out_subj_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(subject_meta, f, indent=2)

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="GFBMD dataset root")
    parser.add_argument("--output-root", required=True, help="Output root for parsing maps")
    parser.add_argument("--sapiens-ckpt", required=True, help="Sapiens TorchScript checkpoint")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLOv8 weights")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES_PER_VIDEO)
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:<index>")
    return parser.parse_args()


if __name__ == "__main__":
    process_dataset(parse_args())
