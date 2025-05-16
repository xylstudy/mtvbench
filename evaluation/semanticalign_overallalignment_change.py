# -*- coding: utf-8 -*-
import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from CLIP.clip import clip
from tqdm import tqdm

# ===================== 配置路径 ===================== #
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
PROMPT_PATH = "/path/to/prompts.txt"
VIDEO_DIR = "/path/to/videos"
OUTPUT_JSON = "/path/to/output/change-alignment_check.json"

# ===================== 加载 CLIP 模型 ===================== #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ===================== 提示词预处理 ===================== #
def process_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    groups = content.split('###')
    return [
        [line.strip() for line in group.strip().split('\n') if line.strip()]
        for group in groups if group.strip()
    ]

# ===================== 视频与提示词绑定 ===================== #
def create_video_prompt_dicts(video_paths, prompt_groups):
    return [{video_paths[i]: prompt_groups[i]} for i in range(min(len(video_paths), len(prompt_groups)))]

# ===================== 提取并平均分割视频帧特征 ===================== #
def extract_segment_features(video_path, num_segments):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_size = total_frames // num_segments
    all_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image_tensor)
        all_features.append(feat)
    cap.release()

    segment_features = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else len(all_features)
        segment = all_features[start:end]
        if segment:
            avg_feat = sum(segment) / len(segment)
            segment_features.append(avg_feat)
    return segment_features

def main():
    prompt_groups = process_prompts(PROMPT_PATH)
    video_paths = [os.path.join(VIDEO_DIR, f"V_{i+81}.mp4") for i in range(len(prompt_groups))]
    video_prompt_pairs = create_video_prompt_dicts(video_paths, prompt_groups)

    change_alignment_score = []

    for video_prompt in video_prompt_pairs:
        video_path = list(video_prompt.keys())[0]
        prompts = list(video_prompt.values())[0]
        print(f"Processing: {video_path}")

        video_segment_features = extract_segment_features(video_path, len(prompts))
        tokenized_prompts = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokenized_prompts)

        video_diff = [
            video_segment_features[i + 1] - video_segment_features[i]
            for i in range(len(video_segment_features) - 1)
        ]
        text_diff = [
            text_features[i + 1] - text_features[i]
            for i in range(len(text_features) - 1)
        ]

        score = sum(F.cosine_similarity(text_diff[i], video_diff[i], dim=-1)
                    for i in range(len(text_diff))) / len(text_diff)
        print(f"Change alignment score: {score.item()}")
        change_alignment_score.append(score.item())

    # 写入结果
    results = {str(i): score for i, score in enumerate(change_alignment_score)}
    results["change_alignment_score"] = sum(change_alignment_score) / len(change_alignment_score)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"结果已保存至：{OUTPUT_JSON}")


if __name__ == "__main__":
    main()
