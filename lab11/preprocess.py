import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

# 配置
DATASET_PATH = '/home/a503218231/cv-course/lab11/archive'
OUTPUT_DIR = 'processed_data'
TARGET_FRAMES = 30
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 引入 MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 创建 PoseLandmarker 实例，使用静态图片模式处理视频帧
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE)  # 逐帧处理用 IMAGE 模式
landmarker = PoseLandmarker.create_from_options(options)

def extract_keypoints(video_path):
    """从视频中提取每帧 132 维关键点向量"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    keypoints_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 用 mediapipe Image 包装帧
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = landmarker.detect(mp_image)
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]  # 取第一个人的关键点
            kp = []
            for lm in landmarks:
                kp.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints_list.append(kp)
        else:
            # 无检测结果时填充 0
            keypoints_list.append([0.0] * 132)
    cap.release()
    if len(keypoints_list) == 0:
        return None
    return np.array(keypoints_list, dtype=np.float32)

def resample_sequence(seq, target_len):
    if len(seq) == 0:
        return np.zeros((target_len, 132), dtype=np.float32)
    if len(seq) == target_len:
        return seq
    old_idx = np.linspace(0, 1, len(seq))
    new_idx = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        resampled[:, j] = np.interp(new_idx, old_idx, seq[:, j])
    return resampled.astype(np.float32)

def normalize_skeleton(seq):
    """以髋部中心为原点，肩宽进行尺度归一化"""
    left_hip = seq[:, 23*4 : 23*4+3]
    right_hip = seq[:, 24*4 : 24*4+3]
    left_shoulder = seq[:, 11*4 : 11*4+3]
    right_shoulder = seq[:, 12*4 : 12*4+3]
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1) + 1e-8
    avg_shoulder_width = np.mean(shoulder_width)
    if avg_shoulder_width == 0:
        avg_shoulder_width = 1.0
    seq_xyz = seq.copy()
    for j in range(33):
        idx = j*4
        seq_xyz[:, idx:idx+3] -= hip_center
        seq_xyz[:, idx:idx+3] /= avg_shoulder_width
    return seq_xyz

def process_dataset():
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)
    print("类别映射:", label_map)

    X, y = [], []
    for cls_name, label in label_map.items():
        cls_dir = os.path.join(DATASET_PATH, cls_name)
        video_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
        print(f"处理类别 '{cls_name}' ({label}), 视频数: {len(video_files)}")
        for vf in tqdm(video_files):
            vp = os.path.join(cls_dir, vf)
            seq = extract_keypoints(vp)
            if seq is None:
                continue
            seq = resample_sequence(seq, TARGET_FRAMES)
            seq = normalize_skeleton(seq)
            X.append(seq)
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"总样本数: {len(X)}, 数据形状: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    print(f"数据已保存到 {OUTPUT_DIR}/")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

if __name__ == '__main__':
    process_dataset()