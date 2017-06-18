from cnn_image_classifier.FileSystemManager import FileSystemManager
from cnn_image_classifier.DownloadManager import DownloadManager
from cnn_image_classifier.cnn_model import train, predict
import logging

logging.basicConfig(level=logging.INFO)
source_archive = 'BreaKHis_v1.tar.gz'
image_directory = 'images'
model_directory = 'tmp'

mode_input = input("Would you like to train (1) the model or use it for a prediction (2): ")

while int(mode_input) not in [1, 2]:
    mode_input = input("Please enter 1 (Train model) or 2 (Prediction): ")

if int(mode_input) == 1:

    clean_run = input("Destroy existing resources and train from scratch " +
                      "(WARNING: this is memory intensive and may take considerable time)? Enter Y/N: ")

    while clean_run.upper() not in ['Y', 'N']:
        clean_run = input("Please enter Y (clean run) or N (retrain existing resources): ")

    if clean_run.upper() == 'Y':

        file_manager = FileSystemManager(image_directory, model_directory)
        file_manager.clean_run()

        download_manager = DownloadManager(source_archive, 'http://www.inf.ufpr.br/vri/databases')
        download_manager.download()

        extract_dir = file_manager.extract_archive(source_archive)
        file_manager.remove_files_except('.png')
        file_manager.data_science_fs(category0='benign', category1='malignant')
        file_manager.organise_files(extract_dir, category_rules={'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'})

    train(image_directory, model_directory)

elif int(mode_input) == 2:

    image_input = input("Would you like to use a randomly selected existing image (1) from our prediction set, " +
                        "or provide your own (2): ")

    while int(image_input) not in [1, 2]:
        image_input = input("Please enter 1 (use existing image) or 2 (own image): ")

    if int(image_input) == 1:
        pass
        # randomly get an image
    elif int(image_input) == 2:
        pass

    prediction, ground_truth = predict(image_directory, model_directory)

    print("Prediction: This is a %s cell.\nValidation: It was a %s cell" % (prediction, ground_truth))
