import csv
import glob
from moviepy.editor import VideoFileClip

# Step 1: List all MP4 files in the folder
video_files = glob.glob("*.mp4")

# Step 2 & 3: Prepare data list
data = [["Video Name", "Duration (Min:Sec)", "Total Frames"]]
total_duration_sec = 0
total_frames = 0

for video_path in video_files:
    clip = VideoFileClip(video_path)
    duration_sec = int(clip.duration)
    frames = int(clip.fps * clip.duration)
    
    # Update totals
    total_duration_sec += duration_sec
    total_frames += frames
    
    # Convert duration to Min:Sec format
    duration_min_sec = f"{duration_sec // 60}:{duration_sec % 60}"
    
    # Append video data
    data.append([video_path.split("/")[-1], duration_min_sec, frames])

# Step 5: Calculate and append totals
total_min_sec = f"{total_duration_sec // 60}:{total_duration_sec % 60}"
data.append(["Total", total_min_sec, total_frames])

# Step 6: Export to CSV
with open("video_info.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)