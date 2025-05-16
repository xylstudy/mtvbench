import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ====================== 路径集中配置 ====================== #
MODEL_PATH = "/data/model/Qwen2_5-VL-7B-Instruct"                      # 预训练模型路径
PROMPT_TXT_PATH = "/data/prompts/prompts.txt"                         # 多段提示词文本（含###分隔）
PROMPT_META_PATH = "/data/prompts/prompts_meta.json"                  # JSON 文件，含每组 prompt 的 action 序列
VIDEO_FRAME_ROOT = "/data/video_frames"                               # 提取帧目录，每个视频为 V_1/, V_2/, ...
OUTPUT_TXT_PATH = "/data/output/action_alignment_scores.txt"          # 输出评分保存路径

VIDEO_FRAME_NUMBER = 144
WINDOW_SIZE = 16
HALF_FRAMES = 48  # 每段时间大约 48 帧

# ====================== 模型加载 ====================== #
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ====================== Prompt 读取 ====================== #
def process_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return [
        [line.strip() for line in group.strip().split('\n') if line.strip()]
        for group in content.split('###') if group.strip()
    ]

prompt_groups = process_prompts(PROMPT_TXT_PATH)
prompt_group_lengths = [len(group) for group in prompt_groups]
num_videos = len(prompt_groups)

# ====================== 视频帧路径构建 ====================== #
video_frames_path_all = []
for i in range(num_videos):
    paths = [
        os.path.join(VIDEO_FRAME_ROOT, f"V_{i+1}", f"V_{i+1}_frame_{j}.jpg")
        for j in range(VIDEO_FRAME_NUMBER)
    ]
    video_frames_path_all.append(paths)

# ====================== 提取时间窗口帧 ====================== #
video_frames_windowed = []
for frames, count in zip(video_frames_path_all, prompt_group_lengths):
    one_video_windows = []
    for i in range(1, count):
        start = int(i * HALF_FRAMES - (WINDOW_SIZE / 2))
        end = int(i * HALF_FRAMES + (WINDOW_SIZE / 2))
        one_video_windows.append(frames[start:end])
    video_frames_windowed.append(one_video_windows)

# ====================== Prompt Metadata 加载 ====================== #
with open(PROMPT_META_PATH, "r", encoding="utf-8") as file:
    prompt_meta_dict = json.load(file)

# ====================== 逐视频打分 ====================== #
action_alignment_score = []
for i in tqdm(range(num_videos), desc="video process"):
    meta = prompt_meta_dict[str(i + 1)]
    actions = [meta.get(f"action{j+1}") for j in range(prompt_group_lengths[i])]
    score_list = []

    for j in range(len(actions) - 1):
        video_clip = [
            Image.open(f).convert("RGB").resize((512, 384), Image.BILINEAR)
            for f in video_frames_windowed[i][j]
        ]
        messages = [
            {"role": "system", "content": "You are an expert in the text-to-video generation field, and now I need your help to answer some questions."},
            {"role": "user", "content": [
                {"type": "video", "video": video_clip},
                {"type": "text", "text": f"Please remember this action: {actions[j]}"},
                {"type": "text", "text": f"Please remember this action: {actions[j+1]}"},
                {"type": "text", "text": (
                    f'When the main subject in a video transitions from "{actions[j]}" to "{actions[j+1]}", '
                    'the transition should fulfill both of the following requirements: '
                    '1.Visual Continuity: The transition between actions should appear smooth and natural, avoiding abrupt cuts or discontinuities. '
                    '2.Logical Consistency: The sequence of actions must follow a realistic and coherent behavioral logic.'
                )},
                {"type": "text", "text": (
                    'Example Illustration: ✅ Appropriate: Transition from “a dog is running” to “a dog is sitting” includes intermediate steps '
                    'where the dog slows down, stops, and then sits. ❌ Inappropriate: A sudden cut from the dog running to the dog sitting '
                    'without any transitional motion or behavioral explanation.'
                )},
                {"type": "text", "text": (
                    'Scoring Guidelines (0 to 3 points): '
                    '0: No transition logic at all. 1: Abrupt and unnatural. 2: Generally logical. 3: Fully coherent and natural.'
                )},
                {"type": "text", "text": "Please only answer with 0, 1, 2, or 3. (without description)"}
            ]}
        ]

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
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        score_list.append(output_text[0])

    action_alignment_score.append(score_list)

# ====================== 保存打分结果 ====================== #
with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
    for row in action_alignment_score:
        f.write(" ".join(map(str, row)) + "\n")

print(f"✅ 分数已成功保存到文件：{OUTPUT_TXT_PATH}")
