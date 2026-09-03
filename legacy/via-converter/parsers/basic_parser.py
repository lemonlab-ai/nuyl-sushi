import os
import jsonpickle


class BaseParser:
    """
    This base class is designed to provide common functionalities for various parser classes.
    It includes methods to write data into a JSON file.
    """

    def __init__(self, output_file: str):
        self.output_file = output_file

    def _write_json_file(self, filename, data):
        """
        Writes the given data into a JSON file specified by filename.
        """
        try:
            with open(filename, "w") as json_file:
                json_data = jsonpickle.encode(
                    data, unpicklable=False, make_refs=False)
                json_file.write(json_data)
            print(f"JSON data written to {filename}")
        except Exception as e:
            print(f"Error writing JSON data: {str(e)}")

    def save_annotation(self, data):
        """
        Saves the annotation data into the output file.
        """
        annotation_data = {
            "videos": data
        }
        folder_path, _ = os.path.split(self.output_file)
        os.makedirs(folder_path, exist_ok=True)
        self._write_json_file(self.output_file, annotation_data)
