# Classifying Cancer
Classifying Cancer is a Python3 project to classify cancer data using
Google's TensorFlow library and Neural Networks.  The goal of this
project was to validate and demonstrate that modern machine learning
techniques in neural nets could prove to be useful in classifying
cancer datasets.

The motivation for applying neural nets at cancer in particular came
from Cancer Research's
[Citizen Science](http://www.cancerresearchuk.org/support-us/citizen-science).
This is a project that relied on volunteers to classify images of breast
cancer tumours. The images themselves contained a mixture of different
looking cells.  Despite having over 2,000,000 contributions, the project
struggled to differentiate cancer cells from non-cancer cells.  Relying
on volunteers to manually classify cancer seemed both inefficient and
ineffective, and I believed that neural nets could provide a better
method for classifying cancer.

This repo contains two main sections:
* [dnn_data_classifier](dnn_data_classifier) - A Deep Neural
Network implementation to classify breast cancer tumours as benign or
malignant depending on measurements taken directly from tumours.
* [cnn_image_classifier](cnn_data_classifier) - A Convolutional
Neural Network implementation to classify breast tumours as benign or
malignant using images of histology slides.

## Getting Started

### Prerequisites

* [Python 3.5+](https://www.python.org/downloads/)

### Installation

1. Install Python3 on your Operating System as per the Python Docs.
Continuum's Anaconda distribution is recommended.

2. Clone the repo:
`git clone https://github.com/jhole89/classifying-cancer.git`

3. Set the environment: `pip install -r requirements.txt`

    (*Optional*: If applicable you can compile Tensorflow for GPU to
    achieve significant performance increases)

### Execution

1. Go to the sub-project directory:
`cd classifying-cancer/dnn_data_classifier` or
`cd classifying-cancer/cnn_image_classifier` depending on whether you
want to classify tumour measurements or tumour images (it is recommended
that you read the relevant README's first).

2. Run the relevant main.py script: `python main.py` or `python
cnn_image_classifier/main.py`.

### Coding style

Classifying-Cancer is PEP8 complaint but uses a max-line-length=100.
This can be checked from the command line with:
```unix
pep8 --statistics --max-line-length=120 classifying-cancer
```

## Built With

* [Python3](https://www.python.org/downloads/)
[(Anaconda)](https://www.continuum.io/downloads)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [TensorFlow](https://www.tensorflow.org)

## Contributing

As I consider this project to be closed I will not be looking to add any
additional features into this project. However if you feel like contributing
then feel free to issue Pull Requests. Any further development or Fork
of this project is bound by the sample license of its parent.

## Authors

* **Joel Lutman**

## License

This project is licensed under the GNU GPL3 License - see the
[LICENSE](LICENSE) file for details

## Acknowledgments

This project makes use of two core data sets for our model training.
Both are brilliant resources for machine learning and I highly suggest
reading the relevant papers listed.

* [UCI's Breast Cancer Wisconsin Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
* [Breast Cancer Histopathological Database](https://web.inf.ufpr.br/vri/breast-cancer-database)
