import os


class ActivityNetAnnotation:
    """
    This class is responsible for creating an annotation from given parameters.
    It provides methods to get the duration, resolution, URL, subset, and annotations of a video file.
    """

    def __init__(self, parent_label: str, filename: str, label: str, coordinates: list, video_info: tuple):
        self.filename = filename
        self.label = parent_label
        self.duration = video_info[0] if video_info is not None else 0
        self.resolution = video_info[1] if video_info is not None else ''
        self.url = video_info[2] if video_info is not None else ''
        self.frames = video_info[3] if video_info is not None else 0
        self.subset = 'training'
        self.annotations = self._get_annotations(coordinates, label)

    def _get_annotations(self, coordinates, label):
        return [{'segment': coordinates, 'label': label}]

    def add_annotation(self, coordinates, label):
        self.annotations.extend(self._get_annotations(coordinates, label))
