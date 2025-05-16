# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def calculate_warp_error(frame1, frame2):
    """计算两帧之间的 warp error（光流重建误差）"""
    if frame1 is None or frame2 is None:
        print("输入帧为 None")
        return 0.0

    if frame1.shape != frame2.shape:
        print("图像尺寸不一致:", frame1.shape, frame2.shape)
        return 0.0

    # 转灰度图
    if frame1.ndim == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if frame2.ndim == 3:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if frame1.ndim != 2 or frame2.ndim != 2:
        print("图像通道数异常:", frame1.ndim, frame2.ndim)
        return 0.0

    try:
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
    except cv2.error as e:
        print("光流计算失败:", e)
        return 0.0

    h, w = frame1.shape
    flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
    flow_map_x = (flow_map_x + flow[..., 0]).astype(np.float32)
    flow_map_y = (flow_map_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(frame2, flow_map_x, flow_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    diff = cv2.absdiff(frame1, warped)
    mask = frame1 > 0

    return float(np.mean(np.square(diff[mask]))) if np.count_nonzero(mask) > 0 else 0.0


def main():
    # ========== 参数配置 ==========
    NUM_VIDEOS = 70
    NUM_FRAMES_PER_VIDEO = 10
    NUM_PROMPTS = 3
    FRAME_DIR_ROOT = "/path/to/video_frames"
    OUTPUT_PATH = "/path/to/output/warp_error_mask_back_10_gdino_1.txt"

    # ========== 构造视频帧路径 ==========
    all_video_frame_paths = []
    for _ in range(1, NUM_PROMPTS):  # prompt index（从 1 到 NUM_PROMPTS-1）
        for vid_idx in range(NUM_VIDEOS):
            frame_paths = [
                os.path.join(FRAME_DIR_ROOT, f"v_{vid_idx}", f"f_{j}_inside_black.jpg")
                for j in range(NUM_FRAMES_PER_VIDEO)
            ]
            all_video_frame_paths.append(frame_paths)

    # ========== 计算 warp error ==========
    all_video_errors = []
    average_errors_per_video = []

    for vid_idx, frame_list in enumerate(all_video_frame_paths):
        video_errors = []
        for j in range(NUM_FRAMES_PER_VIDEO - 1):
            frame1 = cv2.imread(frame_list[j])
            frame2 = cv2.imread(frame_list[j + 1])

            if frame1 is not None and frame2 is not None:
                mse = calculate_warp_error(frame1, frame2)
                print(f"[Video {vid_idx}] Frame {j}-{j+1}: Warp error = {mse:.4f}")
            else:
                print(f"读取失败：{frame_list[j]} 或 {frame_list[j + 1]}")
                mse = 0.0

            video_errors.append(mse)

        avg_video_error = sum(video_errors) / len(video_errors) if video_errors else 0.0
        all_video_errors.append(video_errors)
        average_errors_per_video.append(avg_video_error)

    overall_avg_error = sum(average_errors_per_video) / len(average_errors_per_video) if average_errors_per_video else 0.0

    # ========== 保存结果 ==========
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for errors in all_video_errors:
            f.write(f"{errors}\n")
        f.write(f"average_warp_error: {average_errors_per_video}\n")
        f.write(f"overall_average_error: {overall_avg_error:.6f}\n")

    print("Warp error 计算完成，结果已保存至：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
