# Sleep-Apnea-Detection-1D-CNN
A Deep Learning project using 1D CNN to classify sleep apnea from physiological signals using Leave-One-Participant-Out Cross-Validation


About the Project

This project involves building a machine learning pipeline to detect breathing irregularities (Sleep Apnea) during sleep. It processes physiological signals such as Nasal Airflow, SpO₂, and Thoracic effort to classify 30-second windows as either 'Normal' or 'Apnea'.

To ensure the model is robust and generalizable to new, unseen patients, I implemented a Leave-One-Participant-Out Cross-Validation (LOOCV) strategy using a 1D Convolutional Neural Network (CNN).

Key Achievement: Achieved an overall 88% Accuracy on unseen participant data!

Dataset Details

Source: Continuous text files containing physiological signal data from 5 different participants (AP01 to AP05).

Processing: Applied bandpass digital filters (0.17 Hz to 0.4 Hz) using scipy.signal to remove noise from Flow and Thoracic signals.

Segmentation: Extracted 30-second overlapping windows (120 data points per window).

Final Output: Generated a master dataset containing over 8,800 windows saved in Pickle (.pkl) format to preserve the 3D matrix structure required for deep learning.

Project Structure

The repository contains three main Python scripts:

vis.py: Generates PDF visualizations of the physiological signals and apnea events for initial data exploration.

create_dataset.py: The robust data engineering script that parses raw text files, aligns timestamps, applies digital filters, handles missing data, and generates the final segmented dataset.

train_model.py: Builds and trains the 1D CNN model using Keras/TensorFlow. It executes the LOOCV loop and calculates classification metrics (Accuracy, Precision, Recall, and Confusion Matrix).

How to Run

1. Create the master dataset:
python create_dataset.py -in_dir "Data" -out_dir "Dataset"
2. Train the CNN model:
python train_model.py

Model Architecture (1D CNN)
The deep learning model is built using TensorFlow/Keras and consists of:

Two 1D Convolutional Layers (32 and 64 filters) with ReLU activation for feature extraction.

MaxPooling1D layers to reduce spatial dimensions.

Flatten and Dense layers (64 units) with a 50% Dropout rate to prevent overfitting.

Acknowledgements
AI assistance was used solely for writing the data parsing and preprocessing scripts to extract 30-second windows and build the dataset. The deep learning architecture (1D CNN), LOOCV implementation, and visualizations were developed independently.
