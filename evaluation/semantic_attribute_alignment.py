# -*- coding: utf-8 -*-
import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ====================== 路径配置 ====================== #
BASE_MODEL_PATH = "/path/to/model"
PROMPT_TXT_PATH = "/path/to/prompts.txt"
PROMPT_META_PATH = "/path/to/prompts_meta.json"
VIDEO_PATH_ROOT = "/path/to/videos"
FRAME_PATH_ROOT = "/path/to/video_frames"
OUTPUT_PATH = "/path/to/output/scene_scores.txt"

# ====================== 提示词加载函数 ====================== #
def load_prompt_groups(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    groups = content.split('###')
    return [
        [line.strip() for line in group.strip().split('\n') if line.strip()]
        for group in groups if group.strip()
    ]

# ====================== 视频帧路径构建 ====================== #
def get_frame_paths(frame_root, group_idx, num_frames=121, select_indices=None):
    if select_indices is None:
        select_indices = [i * 3 + 36 for i in range(16)]  # 默认中间 16 帧
    base = f"V_{group_idx + 31}"
    all_paths = [f"{frame_root}/{base}/{base}_frame_{j}.jpg" for j in range(num_frames)]
    return [all_paths[idx] for idx in select_indices]

# ====================== 图像读取和缩放 ====================== #
def load_and_resize_images(image_paths, size=(512, 384)):
    resized = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
        resized.append(img)
    return resized

# ====================== 主流程 ====================== #
def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

    prompt_groups = load_prompt_groups(PROMPT_TXT_PATH)
    prompt_meta = json.load(open(PROMPT_META_PATH, "r", encoding="utf-8"))
    group_lengths = [len(group) for group in prompt_groups]

    video_paths = [f"{VIDEO_PATH_ROOT}/video_{i+30}.mp4" for i in range(len(prompt_groups))]

    all_resized_video_frames = []
    for i in range(len(prompt_groups)):
        selected_paths = get_frame_paths(FRAME_PATH_ROOT, i)
        resized_images = load_and_resize_images(selected_paths)
        all_resized_video_frames.append(resized_images)

    scene_scores = []
    for i in tqdm(range(len(prompt_groups)), desc="Processing videos"):
        scene1 = prompt_meta[str(i + 1)]["scene1"]
        scene2 = prompt_meta[str(i + 1)]["scene2"]
        scene_list = [scene1, scene2]
        print(f"Scene list {i}: {scene_list}")

        scores_per_video = []
        for j in tqdm(range(group_lengths[i] - 1), desc=f"Scoring transitions for video {i}", leave=False):
            messages = [
                {"role": "system", "content": "You are an expert in the text-to-video generation field."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": all_resized_video_frames[i]},
                        {"type": "text", "text": f"Please remember this scene: {scene_list[j]}"},
                        {"type": "text", "text": f"Please remember this scene: {scene_list[j + 1]}"},
                        {"type": "text", "text": (
                            f"Please employ a four-tier quantitative evaluation framework based on semantic congruence: "
                            f"0 = irrelevant, 1 = weak relevance, 2 = mostly accurate, 3 = perfect match between {scene_list[j]} and {scene_list[j+1]}"
                        )},
                        {"type": "text", "text": f"Score the transition from {scene_list[j]} to {scene_list[j+1]}."},
                        {"type": "text", "text": "Please only answer with 0, 1, 2, or 3. (no explanation)"},
                    ]
                }
            ]

            with torch.no_grad():
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                outputs = processor.batch_decode(
                    trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            scores_per_video.append(outputs[0])  # 输出是列表，取第一个回答

        scene_scores.append(scores_per_video)

    # 保存输出结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in scene_scores:
            f.write(" ".join(map(str, row)) + "\n")

    print(f"\n✅ 场景评分已保存至：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
