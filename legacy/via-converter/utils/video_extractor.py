import os
from moviepy.editor import VideoFileClip


class VideoExtractor:
    """
    This class is responsible for extracting information from a video file.
    It provides a method to get the duration, size, and URL of a video file.
    """

    def __init__(self, folder: str, filename: str):
        self.folder = folder
        self.filename = filename

    def get_info(self):
        try:
            url = os.path.normpath(os.path.join(
                self.folder or os.getcwd(), self.filename))
            if os.path.exists(url):
                video = VideoFileClip(url)
                total_frames = int(video.fps * video.duration)
                return (video.duration, video.size, url, total_frames)
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
