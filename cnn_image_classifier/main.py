from cnn_image_classifier.FileSystemManager import FileSystemManager
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

file_manager = FileSystemManager(
    source_data='BreaKHis_v1.tar.gz',
    download_url='http://www.inf.ufpr.br/vri/databases',
    model_dir='',
    source_dir='images'
)

file_manager.clean_run()
file_manager.download()
file_manager.extract()
file_manager.clean_files('.png')
file_manager.flatten()
