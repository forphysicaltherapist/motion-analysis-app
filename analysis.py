import cv2
import mediapipe as mp
import numpy as np
import subprocess

mp_pose = mp.solutions.pose

def get_video_rotation(video_path):
    """ 動画の回転情報を取得（FFmpegを使用） """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        rotation = subprocess.check_output(cmd).decode().strip()
        return int(rotation) if rotation else 0
    except Exception:
        return 0  # 取得できない場合は0（回転なし）

def fix_video_rotation(frame, rotation):
    """ 動画の回転を適用する（FFmpegの回転情報に基づく） """
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame  # 回転なし

def process_video_with_markers(video_path, output_path):
    """ 動画に関節マーカーを描画しながら解析（回転補正あり） """
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    angles_data = []

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # **動画の回転情報を取得**
    rotation = get_video_rotation(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = fix_video_rotation(frame, rotation)  # **回転補正を適用**
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # **体幹の基準線（肩の中点 → 腰の中点）**
            shoulder_mid = [
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2 * frame_width,
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2 * frame_height
            ]
            
            hip_mid = [
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2 * frame_width,
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * frame_height
            ]

            # **肩 - 腰の基準ベクトル**
            torso_vector = np.array([shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]])

            # **右肩 - 右肘の移動ベクトル**
            shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_height
            ]
            elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame_height
            ]
            arm_vector = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])

            # **肩関節の角度計算**
            angle = np.degrees(np.arccos(np.dot(arm_vector, torso_vector) / (np.linalg.norm(arm_vector) * np.linalg.norm(torso_vector))))
            angles_data.append(angle)

            # **関節マーカー描画**
            for point in [shoulder, elbow, shoulder_mid, hip_mid]:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

            # **関節を線で結ぶ**
            cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 255), 2)
            cv2.line(frame, (int(shoulder_mid[0]), int(shoulder_mid[1])), (int(hip_mid[0]), int(hip_mid[1])), (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return angles_data

def get_maximum_range_of_motion(angles_data):
    """ 最大可動域（ROM）を算出 """
    if len(angles_data) == 0:
        return 0
    return max(angles_data) - min(angles_data)
