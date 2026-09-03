import os
import shutil
import argparse
import random
import string
import json

from utils.csv_reader import ViaCSVReader
from utils.video_extractor import VideoExtractor
from utils.video_processor import VideoProcessor

from formats.activitynet.annotation import ActivityNetAnnotation
from formats.nuyl_sushi.annotation import NUYLSushiAnnotation

from parsers.basic_parser import BaseParser
from parsers.activitynet_parser import ActivityNetParser
from parsers.kinectics_parser import KineticsParser


def _generate_next_video_id(target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    existing_ids = set(os.listdir(target_folder))
    while True:
        new_id = ''.join(random.choices(
            string.ascii_letters + string.digits, k=11))
        if new_id not in existing_ids:
            return new_id


def _copy_and_rename_video(filename, source_folder, target_folder="./dataset/encode_videos", log_file="./dataset/filename_log.txt"):
    source_path = os.path.normpath(os.path.join(
        source_folder, os.path.basename(filename)))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    video_id = _generate_next_video_id(target_folder)
    target_path = os.path.normpath(os.path.join(
        target_folder, f"{video_id}.mp4"))

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f"Video copied and renamed to {target_path}")

        with open(log_file, 'a') as log:
            log.write(f"Original: {filename}, New: {target_path}\n")

    return target_path


def _get_view(filename):
    video_filenames_path = 'video_filenames.json'

    # Initialize video_filenames dictionary with both views
    video_filenames = {"front_view": [], "side_view": []}

    # Load the current video filenames from JSON if it exists
    if os.path.exists(video_filenames_path):
        with open(video_filenames_path, 'r') as file:
            video_filenames = json.load(file)

    # Determine the view based on whether the filename is in side_view
    view = 'side_view' if filename in video_filenames.get(
        'side_view', []) else 'front_view'

    return view


def main(csv_folder: str, video_folder: str, output_folder):
    """
    This is the main function that orchestrates the process of reading CSV files from VIA Annotation Tool,
    extracting video information, creating annotations, parsing the data into the specified format,
    and writing the data into a JSON file.

    Parameters:
    csv_folder (str): The path to the folder containing the CSV files.
    video_folder (str): The path to the folder containing the video files.
    output_filepath (str): The path to the output file.
    parser_type (str): The type of parser to use ('activitynet' or 'basic').
    """

    annotations = []

    for parent_label, filename, label, coordinates in ViaCSVReader(csv_folder):

        annotation = next(
            (anno for anno in annotations if anno.filename == filename), None)

        if parent_label == '1_CUTTING-SALMON':
            print("Found")

        if annotation is None:
            video_extractor = VideoExtractor(video_folder, filename)
            video_info = video_extractor.get_info()
            annotations.append(ActivityNetAnnotation(
                parent_label, filename, label, coordinates, video_info))
        else:
            annotation.add_annotation(coordinates, label)

    target_folder = './dataset/encode_videos'
    for annotation in annotations:
        copied_path = _copy_and_rename_video(
            annotation.filename, video_folder, target_folder)
        annotation.url = copied_path
        annotation.view = _get_view(
            annotation.filename)
        annotation.filename = os.path.basename(copied_path)

    if len(annotations) > 0:
        activitynet_path = output_folder + '/activitynet/annotations.json'
        kinectics_path = output_folder + '/kinetics/'
        nuylsushi_path = output_folder + '/nuylsushi/annotations.json'

        # ActivityNet
        parser = ActivityNetParser(activitynet_path, annotations)
        parser.write_json_data()

        # Kinetics
        parser = KineticsParser(kinectics_path, annotations, target_folder)
        parser.save_annotation()

        # NUYLSushi
        video_processor = VideoProcessor()
        sushi_annotations = []
        for annotation in annotations:
            sushi_annotations.append(
                NUYLSushiAnnotation(annotation, video_processor))

        parser = BaseParser(nuylsushi_path)
        parser.save_annotation(sushi_annotations)
    else:
        print("No annotation data found or an error occurred.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', default='./dataset/annotations/via_annotations',
                        help='Path to the folder containing CSV files.')
    parser.add_argument('--video-folder', default='./dataset/videos',
                        help='Path to the folder containing video files.')
    parser.add_argument('--output-folder', default='./output',
                        help='Path to the folder for storing output files.')

    args = parser.parse_args()

    main(args.csv_folder, args.video_folder,
         args.output_folder)
