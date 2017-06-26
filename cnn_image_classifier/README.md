# DNN Cancer Classifier

This section demonstrates a Convolutional Neural Network implementation to
classify breast cancer tumours as benign or malignant depending on
histography images.

## Usage

1. Change directory to the dnn module:
    ```
    $ cd cnn_image_classifier
    ```

2. Run the application:
    ```
    $ python main.py
    ```

### Training

3. The user is given the option to train the model or use it for a prediction.
 Selecting 1 will train the model either from scratch or to carry on training
 an existing model (if one exists) with new epochs. Here we demonstrate a
 completely clean run, where any existing model is removed and a new model
 is trained from scratch.  As with the dnn classifier, the parameters for
 training (epochs, steps, model directory, etc) can be altered in
 cnn_data_classifier/main.py and cnn_data_classifier/cnn_model.py.
 The expected output should be:

    ```
    Would you like to train (1) the model or use it for a prediction (2): 1
    Destroy existing resources and train from scratch (WARNING: this is memory intensive and may take considerable time)? Enter Y/N: y
    INFO:root:Removing resource: Directory [classifying-cancer/cnn_image_classifier/tmp].
    INFO:root:Removing resource: Directory [classifying-cancer/cnn_image_classifier/images].
    INFO:root:Extracting archive BreaKHis_v1.tar.gz to classifying-cancer/cnn_image_classifier/BreaKHis_v1
    INFO:root:Creating resource: Directory [classifying-cancer/cnn_image_classifier/images/train/benign]
    INFO:root:Creating resource: Directory [classifying-cancer/cnn_image_classifier/images/train/malignant]
    INFO:root:Creating resource: Directory [classifying-cancer/cnn_image_classifier/images/predict/benign]
    INFO:root:Creating resource: Directory [classifying-cancer/cnn_image_classifier/images/predict/malignant]
    INFO:root:Removing resource: Directory [classifying-cancer/cnn_image_classifier/BreaKHis_v1].
    INFO:root:Loading resource: Images [classifying-cancer/cnn_image_classifier/images/train]
    WARNING:root:Resource not found: CNN Model [classifying-cancer/cnn_image_classifier/tmp/tensorflow/cnn/model/model.ckpt]. Model will now be trained from scratch.
    INFO:root:Epoch 0 --- Accuracy:  88.3%, Validation Loss: 0.405
    ```

    *(Optional)*: As this process can take a long time depending on local
    resources of the CPU/GPU it can be helpful to visualise the training
    rate, accuracy, and cost.  This can be achieved using TensorBoard,
    which comes packed with TensorFlow.  To activate TensorBoard run:

    ```
    tensorboard --logdir=cnn_image_classifier/tmp/tensorflow/cnn/logs/cnn_with_summaries
    ```
    
    Note: this is the default log directory as defined in main.py, by
    the model_dir property.

    TensorBoard can now be accessed via a web browser at 127.0.0.1:6006
    where a number of metrics can be observed:
    * Scalars: interactive graphs to analyse the accuracy and validation
    loss/cost of the model.
    * Images: displays images currently being processed, reshaped to 64x64 pixels

        <img src="docs/screenshots/TB_images.png" width="600">

    * Graphs: interactive graph to explore the cnn model architecture

        <img src="docs/screenshots/TB_graphs.png" width="600">

    * Distributions: interactive graphs to analyse the distributions of
    weights and biases in the fully connected layers.

    * Histograms: interactive graphs to analyse the percentiles of weights
    and biases in fully connected layers.

### Prediction
