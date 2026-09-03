import os
import cv2

def calculate_frame_count(video_path):
    video_capture =cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()
    return frame_count

def calculate_folder_frame_count(folder_path):
    total_frame_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv')):
                video_path = os.path.join(root, file)
                frame_count = calculate_frame_count(video_path)
                total_frame_count += frame_count
                print(f"{file}: {frame_count} frames")
    # print(f"Total frames in folder: {total_frame_count}")
    return total_frame_count

