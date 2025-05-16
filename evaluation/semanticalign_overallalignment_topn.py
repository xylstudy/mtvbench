# -*- coding: utf-8 -*-
import os
import json
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from CLIP.clip import clip

# 设置镜像源（用于 HF 下载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ===================== 模型初始化 ===================== #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ===================== 提示词加载 ===================== #
def process_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    groups = content.split('###')
    return [
        list(filter(bool, map(str.strip, group.split('\n'))))
        for group in groups if group.strip()
    ]

# ===================== 视频-提示词绑定 ===================== #
def create_video_prompt_dicts(video_paths, prompt_groups):
    return [{video_paths[i]: prompt_groups[i]} for i in range(min(len(video_paths), len(prompt_groups)))]

# ===================== 提取视频帧特征 ===================== #
def extract_video_features(video_path, model, preprocess):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
        features.append(feat)
    cap.release()
    return features

# ===================== 视频帧分段特征（平均） ===================== #
def segment_video_features(video_features, num_segments, start_list, end_list):
    return [
        sum(video_features[start:end]) / len(video_features[start:end])
        for start, end in zip(start_list, end_list)
        if len(video_features[start:end]) > 0
    ]

# ===================== 文本特征提取 ===================== #
def extract_prompt_features(prompts, model):
    tokenized = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        return model.encode_text(tokenized)

# ===================== 相似度矩阵计算（逐点余弦相似度） ===================== #
def compute_similarity_matrix(video_features, prompt_features):
    sim_matrix = torch.zeros(len(prompt_features), len(video_features))
    for i, text_feat in enumerate(prompt_features):
        for j, vid_feat in enumerate(video_features):
            sim_matrix[i, j] = F.cosine_similarity(text_feat, vid_feat, dim=0).item()
    return sim_matrix

# ===================== Top-n 位置对齐度计算 ===================== #
def compute_max_alignment_score(similarity_matrix):
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
    n = similarity_matrix.shape[0]
    threshold = np.partition(similarity_matrix.flatten(), -n)[-n]
    diagonal = similarity_matrix.diagonal()
    return float(np.sum(diagonal >= threshold) / n)

# ===================== 保存结果 ===================== #
def save_alignment_scores(video_names, alignment_scores, output_path, average_score):
    result = {
        str(i): float(alignment_scores[i]) for i in range(len(video_names))
    }
    result["max_alignment_score"] = float(average_score)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# ===================== 主流程 ===================== #
def main():
    # 路径配置（占位形式）
    PROMPT_PATH = "/path/to/prompts.txt"
    VIDEO_DIR = "/path/to/videos"
    OUTPUT_JSON = "/path/to/output/max-alignment_top_check.json"

    # 抽帧区间配置
    start_list = [0, 75]
    end_list = [75, 150]

    # 数据准备
    prompt_groups = process_prompts(PROMPT_PATH)
    video_paths = [os.path.join(VIDEO_DIR, f"V_{i+81}.mp4") for i in range(len(prompt_groups))]
    video_prompt_pairs = create_video_prompt_dicts(video_paths, prompt_groups)
    video_names = [os.path.basename(p) for p in video_paths]

    # 逐视频计算最大对齐分数
    alignment_scores = []
    for pair in video_prompt_pairs:
        video_path = list(pair.keys())[0]
        prompts = list(pair.values())[0]
        print(f"Processing: {video_path}")

        video_feats = extract_video_features(video_path, model, preprocess)
        segmented_video_feats = segment_video_features(video_feats, len(prompts), start_list, end_list)
        text_feats = extract_prompt_features(prompts, model)
        sim_matrix = compute_similarity_matrix(segmented_video_feats, text_feats)
        score = compute_max_alignment_score(sim_matrix)
        alignment_scores.append(score)

    avg_score = sum(alignment_scores) / len(alignment_scores)
    save_alignment_scores(video_names, alignment_scores, OUTPUT_JSON, avg_score)
    print(f"结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
