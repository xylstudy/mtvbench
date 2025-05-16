# -*- coding: utf-8 -*-
import os
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import ViTModel

# 设置 Hugging Face 镜像源（如有需要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 模型加载 ==================== #
MODEL_NAME = "facebook/dino-vitb16"
DINO_MODEL = ViTModel.from_pretrained(
    MODEL_NAME,
    cache_dir="/path/to/cache/dino",
    ignore_mismatched_sizes=True,
    force_download=False,
).to("cuda" if torch.cuda.is_available() else "cpu").eval()

# ==================== 图像预处理 ==================== #
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==================== 提示词处理 ==================== #
def process_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    groups = content.split('###')
    return [
        [line.strip() for line in group.strip().split('\n') if line.strip()]
        for group in groups if group.strip()
    ]

# ==================== 视频特征提取 ==================== #
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    features = []
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tensor = preprocess(frame).unsqueeze(0).to(DINO_MODEL.device)
            feat = DINO_MODEL(tensor).last_hidden_state.mean(dim=1).cpu().numpy()
            features.append(feat)
    cap.release()
    return features

# ==================== 分段平均特征 ==================== #
def split_and_average(feat_list, num_segments):
    if num_segments == 0:
        return []

    start_indices = [0, 75]
    end_indices = [75, 150]
    avg_feats = []

    for i in range(num_segments):
        part = feat_list[start_indices[i]:end_indices[i]]
        avg_feats.append(np.mean(part, axis=0) if part else 0)
    return avg_feats

# ==================== 余弦相似度计算（Torch） ==================== #
def cosine_similarity_torch(feat1, feat2):
    t1 = torch.tensor(feat1).view(-1)
    t2 = torch.tensor(feat2).view(-1)
    return F.cosine_similarity(t1, t2, dim=0)

# ==================== 主流程 ==================== #
def main():
    # ---------- 路径配置 ----------
    PROMPT_PATH = "/path/to/prompts.txt"
    VIDEO_DIR = "/path/to/videos"
    OUTPUT_JSON = "/path/to/output/temporal_consistency_object_check.json"

    # ---------- 加载提示词 ----------
    prompt_groups = process_prompts(PROMPT_PATH)
    prompt_counts = [len(p) for p in prompt_groups]
    video_paths = [os.path.join(VIDEO_DIR, f"V_{i+151}.mp4") for i in range(len(prompt_groups))]

    # ---------- 特征提取与相似度计算 ----------
    temporal_consistency_scores = []

    for idx in tqdm(range(len(video_paths)), desc="处理视频"):
        video_path = video_paths[idx]
        prompts = prompt_groups[idx]
        num_segments = len(prompts)

        video_feat = extract_video_features(video_path)
        segment_avg_feat = split_and_average(video_feat, num_segments)

        if not segment_avg_feat:
            temporal_consistency_scores.append(0)
            continue

        start_feat = video_feat[0]
        end_feat = video_feat[-1]

        start_sim = [cosine_similarity_torch(start_feat, seg_feat) for seg_feat in segment_avg_feat]
        end_sim = [cosine_similarity_torch(end_feat, seg_feat) for seg_feat in segment_avg_feat]

        start_avg = sum(start_sim) / len(start_sim)
        end_avg = sum(end_sim) / len(end_sim)

        final_score = (start_avg + end_avg) / 2
        temporal_consistency_scores.append(final_score)

    # ---------- 保存 JSON ----------
    avg_score = sum(temporal_consistency_scores) / len(temporal_consistency_scores)
    result_dict = {
        f"video_{i}": score.item() if isinstance(score, torch.Tensor) else float(score)
        for i, score in enumerate(temporal_consistency_scores)
    }
    result_dict["average_temporal_consistency"] = float(avg_score)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"评分完成，结果已写入：{OUTPUT_JSON}")

if __name__ == "__main__":
    main()
