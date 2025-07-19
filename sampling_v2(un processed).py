import cv2
import mediapipe as mp
import time
import json
import numpy as np
from extends import zb_x, zb_y
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 需要采样的视频名称
video_name = "normal"

video_path = f"input_mp4/{video_name}.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

JOINT_MAPPING = {
    27: "left_ankle",
    25: "left_knee",
    23: "left_hip",
    24: "right_hip",
    26: "right_knee",
    28: "right_ankle",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    0: "nose"
}

output_data = {
    "video_info": {
        "fps": float(fps),
        "total_frames": total_frames,
        "resolution": [height, width]
    },
    "frames": []
}


def calculate_special_points(landmarks, frame_shape):
    h, w = frame_shape[:2]

    pelvis_x = (landmarks[23].x + landmarks[24].x) / 2 * w
    pelvis_y = (landmarks[23].y + landmarks[24].y) / 2 * h

    thorax_x = (landmarks[11].x + landmarks[12].x) / 2 * w
    thorax_y = (landmarks[11].y + landmarks[12].y) / 2 * h

    upper_neck_x = (thorax_x + landmarks[0].x * w) / 2
    upper_neck_y = (thorax_y + landmarks[0].y * h) / 2

    head_top_x = landmarks[0].x * w
    head_top_y = landmarks[0].y * h - 0.125 * h  # 上移12.5%画面高度

    return {
        "pelvis": (pelvis_x, pelvis_y),
        "thorax": (thorax_x, thorax_y),
        "upper_neck": (upper_neck_x, upper_neck_y),
        "head_top": (head_top_x, head_top_y)
    }


try:
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        frame_info = {
            "frame_number": frame_count,
            "timestamp": frame_count / fps,
            "joints": {}
        }

        special_points = calculate_special_points(
            results.pose_landmarks.landmark,
            frame.shape
        )

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in JOINT_MAPPING:
                joint_name = JOINT_MAPPING[idx]
                frame_info["joints"][joint_name] = {
                    "x": float(landmark.x * width),
                    "y": float(landmark.y * height),
                    "confidence": 1.0
                }

        for point_name, (x, y) in special_points.items():
            frame_info["joints"][point_name] = {
                "x": float(x),
                "y": float(y),
                "confidence": 1.0
            }

        output_data["frames"].append(frame_info)
        frame_count += 1

except Exception as e:
    print(f"处理出错: {str(e)}")
finally:
    cap.release()
    with open(f"{video_name}.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("处理完成，数据已保存")