# VIA to ActivityNet Annotation Converter

This project is a script for converting annotations from the VIA Annotation Tools format to the ActivityNet annotation format.

## Project Structure

```
.
├── main.py
├── models
│   ├── annotation.py
│   └── taxonomy.py
├── output
│   ├── annotation.json
│   └── taxonomy.json
├── parsers
│   ├── activity_net_parser.py
├── utils
│   ├── csv_reader.py
│   └── video_extractor.py
├── videos
│   ├── ...
├── annotations
│   ├── ....csv
│   └── ....json
├── requirements.txt
```

## Description
- `main.py`: The main script that runs the conversion process.
- `models/`: Contains the Annotation and Taxonomy classes.
- `output/`: The directory where the converted annotation and taxonomy JSON files are saved.
- `parsers/`: Contains the ActivityNetParser class for parsing data into the ActivityNet format.
- `utils/`: Contains utility classes for reading CSV files and extracting video information.
- `videos/`: The directory where the video files are stored.
- `annotations/`: The directory where the VIA annotation CSV and JSON files are stored.
- `requirements.txt`: Contains the Python dependencies required for this project.

## Usage
1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Run the `main.py` script:
```
python main.py
```


## Command-Line Arguments

The `main.py` script accepts the following command-line arguments:

- `--csv-folder`: Path to the folder containing CSV files. Default is `./annotations`.
- `--video-folder`: Path to the folder containing video files. Default is `./videos`.
- `--output-file`: Path to the output file for storing the converted annotations. Default is `./output/annotation.json`.

You can specify these arguments when running the script like this:

```bash
python main.py --csv-folder ./path/to/csvs
--video-folder ./path/to/videos
--output-file ./path/to/output.json