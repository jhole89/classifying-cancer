from cnn_image_classifier.FileSystemManager import FileSystemManager
from cnn_image_classifier.DownloadManager import DownloadManager
from cnn_image_classifier.cnn_model import train
import logging

logging.basicConfig(level=logging.INFO)
source_archive = 'BreaKHis_v1.tar.gz'

file_manager = FileSystemManager('images')
file_manager.clean_run()

download_manager = DownloadManager(source_archive, 'http://www.inf.ufpr.br/vri/databases')
download_manager.download()

extract_dir = file_manager.extract_archive(source_archive)
file_manager.remove_files_except('.png')
file_manager.flatten_directory(extract_dir)

file_index = file_manager.index_directory()
train_list, test_list = file_manager.split_sets(file_index, test_rate=0.2)
train(train_list, test_list)

