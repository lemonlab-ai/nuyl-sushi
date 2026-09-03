import os
import shutil


class KineticsParser:
    def __init__(self, output_folder: str, data: list = None, source_folder: str = "./source_videos") -> None:
        self.output_folder = output_folder
        self.data = data
        self.source_folder = source_folder
        # Define new paths for coarse and fine class folders and list files
        self.coarse_train_folder = os.path.join(
            output_folder, "nuylsushi/coarse/videos_train")
        self.coarse_val_folder = os.path.join(
            output_folder, "nuylsushi/coarse/videos_val")
        self.fine_train_folder = os.path.join(
            output_folder, "nuylsushi/fine/videos_train")
        self.fine_val_folder = os.path.join(
            output_folder, "nuylsushi/fine/videos_val")
        self.coarse_train_list_path = os.path.join(
            output_folder, "nuylsushi/nuylsushi_coarse_train_list_videos.txt")
        self.coarse_val_list_path = os.path.join(
            output_folder, "nuylsushi/nuylsushi_coarse_val_list_videos.txt")
        self.fine_train_list_path = os.path.join(
            output_folder, "nuylsushi/nuylsushi_fine_train_list_videos.txt")
        self.fine_val_list_path = os.path.join(
            output_folder, "nuylsushi/nuylsushi_fine_val_list_videos.txt")

        self.fine_seg_train_list_path = os.path.join(
            output_folder, "nuylsushi/nuylsushi_fine_seg_train_list_videos.txt")
        self.fine_seg_val_list_path = os.path.join(
            output_folder, "nuylsushi/nuylsushi_fine_seg_val_list_videos.txt")

        # Ensure all folders exist
        os.makedirs(self.coarse_train_folder, exist_ok=True)
        os.makedirs(self.coarse_val_folder, exist_ok=True)
        os.makedirs(self.fine_train_folder, exist_ok=True)
        os.makedirs(self.fine_val_folder, exist_ok=True)

    def _generate_class_lists(self):
        """
        Generates dictionaries mapping class names to unique numbers for both coarse and fine classes,
        and also counts the occurrences of each class.
        """
        coarse_class_set = set()
        fine_class_set = set()
        coarse_class_count = {}
        fine_class_count = {}

        for annotation in self.data:
            coarse_class_set.add(annotation.label)
            coarse_class_count[annotation.label] = coarse_class_count.get(
                annotation.label, 0) + 1
            for ann in annotation.annotations:
                fine_class_set.add(ann['label'])
                fine_class_count[ann['label']] = fine_class_count.get(
                    ann['label'], 0) + 1

        coarse_class_list = {classname: index for index,
                             classname in enumerate(sorted(coarse_class_set))}
        fine_class_list = {classname: index for index,
                           classname in enumerate(sorted(fine_class_set))}

        return coarse_class_list, fine_class_list, coarse_class_count, fine_class_count

    def _save_class_counts(self, coarse_class_count, fine_class_count):
        """
        Saves the counts of each coarse and fine class to a log file.
        """
        log_path = os.path.join(self.output_folder, "class_counts_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write("Coarse Class Counts:\n")
            for classname, count in sorted(coarse_class_count.items()):
                log_file.write(f"{classname}: {count}\n")
            log_file.write("\nFine Class Counts:\n")
            for classname, count in sorted(fine_class_count.items()):
                log_file.write(f"{classname}: {count}\n")

    def _save_video_lists(self, coarse_class_list, fine_class_list, split_ratio=0.8):
        """
        Saves the coarse and fine video lists and copies videos to the respective folders.
        Adjusts for both training and validation data based on a split ratio.
        """
        # Calculate split index
        split_index = int(len(self.data) * split_ratio)

        with open(self.coarse_train_list_path, "w") as coarse_train_file, \
                open(self.fine_train_list_path, "w") as fine_train_file, \
                open(self.coarse_val_list_path, "w") as coarse_val_file, \
                open(self.fine_val_list_path, "w") as fine_val_file, \
                open(self.fine_seg_train_list_path, "w") as fine_seg_train_file, \
                open(self.fine_seg_val_list_path, "w") as fine_seg_val_file:

            for i, annotation in enumerate(self.data):
                coarse_class_number = coarse_class_list[annotation.label]
                src_file_path = os.path.join(
                    self.source_folder, annotation.filename)

                if i < split_index:  # Training data
                    shutil.copy(src_file_path, self.coarse_train_folder)
                    coarse_train_file.write(f"{annotation.filename} {
                                            coarse_class_number}\n")
                    for ann in annotation.annotations:
                        fine_class_number = fine_class_list[ann['label']]
                        start_time, end_time = ann['segment']
                        shutil.copy(src_file_path, self.fine_train_folder)

                        fine_train_file.write(f"{annotation.filename} {
                                              fine_class_number}\n")
                        fine_seg_train_file.write(f"{annotation.filename} {start_time} {
                            end_time} {fine_class_number}\n")
                else:  # Validation data
                    shutil.copy(src_file_path, self.coarse_val_folder)
                    coarse_val_file.write(f"{annotation.filename} {
                        coarse_class_number}\n")
                    for ann in annotation.annotations:
                        fine_class_number = fine_class_list[ann['label']]
                        start_time, end_time = ann['segment']
                        shutil.copy(src_file_path, self.fine_val_folder)

                        fine_val_file.write(f"{annotation.filename} {
                                            fine_class_number}\n")
                        fine_seg_val_file.write(f"{annotation.filename} {start_time} {
                            end_time} {fine_class_number}\n")

    def _save_class_lists(self, coarse_class_list, fine_class_list):
        """
        Saves the coarse and fine class lists with class names and numbers.
        """
        coarse_folder_path = self.output_folder + "coarse_class_list.txt"
        fine_folder_path = self.output_folder + "fine_class_list.txt"
        with open(coarse_folder_path, "w") as coarse_file, open(fine_folder_path, "w") as fine_file:
            for classname, number in coarse_class_list.items():
                coarse_file.write(f"{classname} {number}\n")
            for classname, number in fine_class_list.items():
                fine_file.write(f"{classname} {number}\n")

    def save_annotation(self):
        """
        Generates class lists and saves the video and class list files.
        """
        if not self.data:
            print("No data provided.")
            return

        coarse_class_list, fine_class_list, coarse_class_count, fine_class_count = self._generate_class_lists()
        self._save_video_lists(coarse_class_list, fine_class_list)
        self._save_class_lists(coarse_class_list, fine_class_list)
        self._save_class_counts(coarse_class_count, fine_class_count)
