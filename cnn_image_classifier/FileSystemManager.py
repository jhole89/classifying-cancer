import os
import shutil
import logging
import tarfile


class FileSystemManager:

    def __init__(self, source_dir=None, model_dir=None):
        self.source_dir = source_dir
        self.model_dir = model_dir
        self.archive_dir = None

    def clean_run(self):
        """Remove model and data dirs for a clean run"""

        for directory in [self.model_dir, self.source_dir]:
            if directory:
                if os.path.exists(directory):
                    try:
                        logging.info("Removing resource: Directory [%s].", directory)
                        shutil.rmtree(directory)
                    except OSError:
                        logging.error("Could not remove resource: Directory [%s].", directory)

    def extract_archive(self, archive):
        """Extract compressed archives tar.gz"""

        self.archive_dir = archive.split('.')[0]

        if not os.path.exists(self.archive_dir):
            logging.info("Extracting archive %s to %s", archive, self.archive_dir)

            if archive.lower().endswith('.tar.gz'):
                tar = tarfile.open(archive, "r:gz")
            tar.extractall()
            tar.close()

        return self.archive_dir

    def remove_files_except(self, extension):
        """Removes all files not ending in extension"""

        for root, dirs, files in os.walk(self.archive_dir):
            for current_file in files:
                if not current_file.lower().endswith(extension):
                    try:
                        logging.debug("Removing resource: File [%s]", current_file)
                        os.remove(os.path.join(root, current_file))
                    except OSError:
                        logging.error("Could not remove resource: File [%s]", current_file)

    def flatten_directory(self, directory):
        """Flattens directory tree to single level"""

        os.mkdir(self.source_dir)

        for root, dirs, files in os.walk(directory):
            for filename in files:

                try:
                    logging.debug("Moving %s from %s to %s", filename, root, self.source_dir)
                    os.rename(os.path.join(root, filename), os.path.join(self.source_dir, filename))

                except OSError:
                    logging.error("Could not move %s ", os.path.join(root, filename))

        try:
            logging.info("Removing resource: Directory [%s].", self.archive_dir)
            shutil.rmtree(self.archive_dir)
        except OSError:
            logging.error("Could not remove resource: Directory [%s].", self.archive_dir)

    def index_directory(self):
        """Generate index for directory"""

        labels = []

        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                labels.append(file[4])

        return files, labels
