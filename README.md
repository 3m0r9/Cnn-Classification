# Grapevine Leaves Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of grapevine leaves into different grape varieties. The model leverages the power of deep learning to accurately identify the type of grapevine based on the leaf image, helping to automate the process of vine classification.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributors](#contributors)
10. [License](#license)
11. [Let's Connect](#lets-connect)

## Project Overview

Grapevine leaf classification is an important task in viticulture for identifying different grape varieties. Manual classification is time-consuming and requires expert knowledge. This project aims to automate this task using a CNN model that can classify grapevine leaves based on their images. The project involves:

- Preprocessing the dataset of grapevine leaf images.
- Building a CNN model to classify the images.
- Evaluating the model performance on the test set.
- Visualizing the results to understand the model’s predictions.

## Dataset

The dataset used for this project is the [Grapevine Leaves Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/grapevine-leaves-image-dataset) available on Kaggle. It contains images of grapevine leaves from different grape varieties.

- **Classes**: 
  - The dataset includes images from 9 different grapevine cultivars.
- **Total Images**: 
  - 5,000 images.
- **Image Size**: 
  - All images are of size 256x256 pixels.

## Data Preprocessing

Before feeding the images into the CNN model, the following preprocessing steps are applied:

1. **Image Resizing**: All images are resized to a uniform size suitable for the CNN model (e.g., 128x128 pixels).
2. **Normalization**: Pixel values are normalized to a range of [0, 1] to speed up convergence.
3. **Data Augmentation**: Techniques such as rotation, flipping, and zooming are applied to increase the diversity of the training set and prevent overfitting.
4. **Train-Test Split**: The dataset is split into training and test sets, typically with a ratio of 80:20.

## Modeling

### Convolutional Neural Network (CNN)

The CNN architecture is designed to extract hierarchical features from the images and perform classification:

- **Conv2D Layers**: Extract spatial features using filters.
- **MaxPooling2D Layers**: Reduce spatial dimensions and retain important features.
- **Dropout Layers**: Prevent overfitting by randomly setting a fraction of input units to zero during training.
- **Dense Layers**: Fully connected layers to perform the final classification.
- **Activation**: `ReLU` activation is used for hidden layers, and `softmax` for the output layer to predict the probabilities of each class.

### Libraries Used:
- TensorFlow / Keras
- NumPy
- Matplotlib (for visualization)
- OpenCV (for image preprocessing)

## Evaluation

The performance of the CNN model is evaluated using the following metrics:

- **Accuracy**: Percentage of correctly classified images in the test set.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: Visual representation of the true versus predicted classes.

## Installation

To run this project on your local machine, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/3m0r9/Cnn-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Cnn-Classification
   ```
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/grapevine-leaves-image-dataset) and extract it into the `data/` directory.
2. Preprocess the images and create the training and test sets:
   ```bash
   python preprocess_data.py --input data/grapevine_leaves --output data/processed
   ```
3. Train the CNN model:
   ```bash
   python train_model.py --input data/processed
   ```
4. Evaluate the model on the test set:
   ```bash
   python evaluate_model.py --input data/processed/test
   ```

## Results

The CNN model achieved the following results on the test set:

- **Accuracy**: 90%
- **Precision**: 88%
- **Recall**: 87%
- **F1-Score**: 87.5%

### Visualizations:

- **Confusion Matrix**: Displays the true versus predicted labels for the test set.
- **Sample Predictions**: Example images with their predicted and actual labels.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)
