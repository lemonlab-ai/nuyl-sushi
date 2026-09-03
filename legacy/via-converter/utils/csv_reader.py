import csv
import os
import re
from io import StringIO
import jsonpickle


class ViaCSVReader:
    """
    This class is responsible for reading and processing VIA's annotation CSV files from a given folder.
    It provides methods to get a list of CSV files, process the CSV header, and read the CSV files.
    """

    def __init__(self, csv_folder: str):
        self.data = []
        self._read(csv_folder)

    def _read(self, csv_folder: str):
        for csv_file in self._get_csv_list(csv_folder):
            for row in csv.DictReader(StringIO(self._process_csv_header(csv_file))):
                video_filename = jsonpickle.decode(row['file_list'])[0]
                parent_label = self._get_parent_label(csv_file, video_filename)
                label = jsonpickle.decode(row['metadata'])['1']
                if label == '1_CUTTING-SALMON':
                    label = 'CUTTING-SALMON'
                temporal_coordinates = jsonpickle.decode(
                    row['temporal_coordinates'])

                self.data.append(
                    (parent_label, video_filename, label, temporal_coordinates))

    def _get_csv_list(self, csv_folder: str):
        return [os.path.join(csv_folder, filename)
                for filename in os.listdir(csv_folder)
                if filename.endswith('.csv')]

    def _process_csv_header(self, filename: str):
        try:
            with open(filename, 'r') as original_file:
                modified_text = '\n'.join(line.replace('# CSV_HEADER = ', '')
                                          for line in original_file.readlines()
                                          if not (line.startswith('#') and 'CSV_HEADER' not in line))

                return modified_text
        except FileNotFoundError:
            print(f'Error: File not found - {filename}')
            return None
        except Exception as e:
            print(f'Error: {e}')
            return None

    def _get_parent_label(self, csvfile_name: str, video_filename: str):

        video_name, _ = os.path.splitext(video_filename)
        csvfile_name = os.path.basename(csvfile_name)

        result = csvfile_name.replace('.csv', '').replace(video_name + '_', '')

        result = re.sub(r'\(.*\)', '', result)
        result = result.upper().replace(' ', '')

        if result == '1_CUTTING-SALMON':
            result = 'CUTTING-SALMON'

        return result

    def __getitem__(self, index):
        return self.data[index]
