import cv2
import os
import mediapipe as mp
import torch
import numpy as np
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Set paths
VIDEO_FILE = '.\\data\\d1.mp4'
OUTPUT_DIR = '.\\processed\\'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Lip landmark indices in MediaPipe
LIP_LANDMARKS = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324
]))

def extract_lip_roi(frame, landmarks):
    h, w, _ = frame.shape
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LIP_LANDMARKS])
    x, y, w_, h_ = cv2.boundingRect(points)
    lip_roi = frame[y:y+h_, x:x+w_]
    return lip_roi

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(img_rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                lip = extract_lip_roi(frame, landmarks)
                if lip.size == 0:
                    continue
                gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized)

    cap.release()
    if not frames:
        return None
    tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    return tensor.unsqueeze(1)  # Shape: [T, 1, 64, 64]
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(img_rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                lip = extract_lip_roi(frame, landmarks)
                if lip.size == 0:
                    print(f"Skipping empty lip crop in {video_path}")
                    continue
                gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized)

    cap.release()
    if not frames:
        print(f"No frames extracted from {video_path}")
        return None
    tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    return tensor.unsqueeze(1)  # Shape: [T, 1, 64, 64]
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = frames[::5]

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(img_rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                lip = extract_lip_roi(frame, landmarks)
                if lip.size == 0:
                    print(f"Skipping empty lip crop in {video_path}")
                    continue
                gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized)

    cap.release()
    if not frames:
        print(f"No frames extracted from {video_path}")
        return None
    tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    return tensor.unsqueeze(1)  # Shape: [T, 1, 64, 64]
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        if result.multi_face_landmarks:
            print("Face detected")
        else:
            print("No face detected")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(img_rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                lip = extract_lip_roi(frame, landmarks)
                if lip.size == 0:
                    print(f"Skipping empty lip crop in {video_path}")
                    continue
                gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized)

    cap.release()
    if not frames:
        print(f"No frames extracted from {video_path}")
        return None
    tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    return tensor.unsqueeze(1)  # Shape: [T, 1, 64, 64]
def process_video(video_path):
    print(f"Starting video processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(img_rgb)

            if result.multi_face_landmarks:
                print(f"Frame {frame_count}: Face detected")
                landmarks = result.multi_face_landmarks[0].landmark
                lip = extract_lip_roi(frame, landmarks)
                if lip.size == 0:
                    print(f"Frame {frame_count}: Empty lip crop")
                    continue
                gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized)
            else:
                print(f"Frame {frame_count}: No face detected")

    cap.release()
    if not frames:
        print(f"No usable frames extracted from {video_path}")
        return None
    tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0
    return tensor.unsqueeze(1)


def process_all():
    print(f"Processing: {VIDEO_FILE}")
    tensor = process_video(VIDEO_FILE)
    if tensor is not None:
        save_path = os.path.join(OUTPUT_DIR, 'd1.pt')
        print(f"Saving to: {save_path}")
        torch.save(tensor, save_path)
    else:
        print("No frames extracted from the video")



if __name__ == "__main__":
    process_all()
