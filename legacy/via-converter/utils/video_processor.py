import os
from moviepy.editor import VideoFileClip


class VideoProcessor:
    def __init__(self, images_folder="./dataset/images", fps=15, log_file="./dataset/filename_log.txt"):
        self.images_folder = images_folder
        self.fps = fps
        self.log_file = log_file
        self.total_frames = 0

    def process_video(self, filename, annotations):
        video_id = os.path.splitext(os.path.basename(filename))[0]
        image_pairs, total_frames = self.extract_images(
            video_id, filename, annotations)
        self._log_frame_info(filename, video_id,
                             total_frames, len(image_pairs))
        return image_pairs

    def _log_frame_info(self, original_filename, video_id, extracted_frames, current_frames):
        with open(self.log_file, 'a') as log:
            log.write(f"Video ID: {video_id}, Original: {original_filename}, Extracted Frames: {
                      extracted_frames}, Current Frames: {current_frames}\n")

    def extract_images(self, video_id, video_path, annotations):
        images_path = os.path.join(
            self.images_folder, video_id, f"fps{self.fps}")

        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        clip = VideoFileClip(video_path)
        total_frames = int(clip.fps * clip.duration)
        self.total_frames += total_frames
        image_paths = []
        image_times = []

        for annotation in annotations:
            start_time = annotation['start_time']
            end_time = annotation['end_time']
            start_frame = int(start_time * clip.fps)
            end_frame = int(end_time * clip.fps)

            for frame_number in range(start_frame, end_frame, self.fps):
                real_time = frame_number / clip.fps
                if real_time < start_time or real_time > end_time:
                    continue
                img_path = os.path.join(
                    images_path, f"{video_id}_{real_time}.jpg")
                # clip.save_frame(img_path, real_time)
                image_paths.append(img_path)
                image_times.append(real_time)

        image_pairs = list(zip(image_paths, image_times))

        return image_pairs, total_frames
