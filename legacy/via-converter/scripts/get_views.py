import os
import json
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct absolute paths to the folders
front_view_path = os.path.join(script_dir, '..', 'dataset', 'front_view')
side_view_path = os.path.join(script_dir, '..', 'dataset', 'side_view')


def read_video_filenames(folder_path):
    """Read video filenames from a folder."""
    video_filenames = [file for file in os.listdir(
        folder_path) if file.endswith('.mp4')]
    return video_filenames


# Initialize dictionary to store filenames
video_filenames_dict = {}

# Read and store video filenames from both folders
video_filenames_dict['front_view'] = read_video_filenames(front_view_path)
video_filenames_dict['side_view'] = read_video_filenames(side_view_path)

# Print or return the dictionary
print(video_filenames_dict)

with open('video_filenames.json', 'w') as json_file:
    json.dump(video_filenames_dict, json_file, indent=4)
