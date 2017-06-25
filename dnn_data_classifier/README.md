# DNN Cancer Classifier

This section demonstrates a Deep Neural Network implementation to
classify breast cancer tumours as benign or malignant depending on
measurements taken directly from tumours.


## Usage

1. Change directory to the dnn module:
    ```
    $ cd dnn_data_classifier
    ```

2. Run the application:
    ```
    $ python main.py
    ````

3. Any existing model will be removed and a new model will be trained.
The parameters for training (epochs, steps, model directory, etc) can be
altered in dnn_data_classifier/main.py. The expected output should be:
    ```
    Deleting resource: Model directory [nn_classifier].
    Removed resource: Model directory [nn_classifier].
    Deleting resource: Data [training_set.csv].
    Removed resource: Data [training_set.csv].
    Deleting resource: Data [test_set.csv].
    Removed resource: Data [test_set.csv].
    Model trained after 2000 steps.

    Model Accuracy: 0.927007
    ```

4. With the model now trained and fairly accurate, it can be used for
prediction, either by entering your own measurements or by similating
fake data:
    ```
    Predict classification: Enter own data (1) or simulate fake data (2)?
     Enter 1 or 2: 1
    Enter value 0-10 for clump_thickness: 1
    Enter value 0-10 for unif_cell_size: 5
    Enter value 0-10 for unif_cell_shape: 6
    Enter value 0-10 for marg_adhesion: 4
    Enter value 0-10 for single_epith_cell_size: 8
    Enter value 0-10 for bare_nuclei: 2
    Enter value 0-10 for bland_chrom: 6
    Enter value 0-10 for norm_nucleoli: 9
    Enter value 0-10 for mitoses: 3

    Class Prediction: malignant

    Would you like to try another prediction? Enter Y/N: y
    Predict classification: Enter own data (1) or simulate fake data (2)?
     Enter 1 or 2: 2
    Data generated:
    clump_thickness: 0
    unif_cell_size: 1
    unif_cell_shape: 2
    marg_adhesion: 3
    single_epith_cell_size: 4
    bare_nuclei: 5
    bland_chrom: 6
    norm_nucleoli: 7
    mitoses: 8

    Class Prediction: benign
    ```
