from cnn_image_classifier.FileSystemManager import FileSystemManager
from cnn_image_classifier.DownloadManager import DownloadManager
from cnn_image_classifier.cnn_model import train
import logging
logging.basicConfig(level=logging.INFO)
source_archive = 'BreaKHis_v1.tar.gz'
image_directory = 'images'
model_directory = 'tmp'

file_manager = FileSystemManager(image_directory, model_directory)
file_manager.clean_run()

download_manager = DownloadManager(source_archive, 'http://www.inf.ufpr.br/vri/databases')
download_manager.download()

extract_dir = file_manager.extract_archive(source_archive)
file_manager.remove_files_except('.png')
file_manager.data_science_fs(category0='benign', category1='malignant')
file_manager.organise_files(extract_dir, category_rules={'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'})

train(image_directory, model_directory)
