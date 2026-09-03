import os
from formats.activitynet.annotation import ActivityNetAnnotation
from utils.video_processor import VideoProcessor


class NUYLSushiAnnotation:
    def __init__(self, annotation: ActivityNetAnnotation, video_processor: VideoProcessor) -> None:
        self.annotations = self._convert_annotations(annotation.annotations)
        self.category = annotation.label
        self.view_type = annotation.view
        self.gyroscope_data = "path/to/gyroscope_data.csv"
        self.weight_g = []

        self.image_sampling = self._create_image_sampling(
            annotation.url, self.annotations, video_processor)

    def _convert_annotations(self, annotations):
        converted_annotations = []
        for ann in annotations:
            converted_annotation = {
                "start_time": ann['segment'][0],
                "end_time": ann['segment'][1],
                "action": ann['label']
            }
            converted_annotations.append(converted_annotation)
        return converted_annotations

    def _create_image_sampling(self, filename, annotations, video_processor):
        image_text_pairs = []
        image_pairs = video_processor.process_video(filename, annotations)
        for path, t in image_pairs:
            text_description = self._get_text_description_for_time(t)
            image_text_pairs.append({
                "image_path": os.path.normpath(path),
                "text_description": text_description
            })
        return {
            "default_sampling_rate": video_processor.fps,
            "image_text_pairs": image_text_pairs
        }

    def _get_text_description_for_time(self, t):
        # Find the annotation that covers the time t and return its label/action
        for ann in self.annotations:
            if ann['start_time'] <= t <= ann['end_time']:
                return ann['action']
        return ""
