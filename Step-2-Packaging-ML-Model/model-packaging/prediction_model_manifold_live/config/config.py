import pathlib
import os
import prediction_model_manifold_live

PACKAGE_ROOT = pathlib.Path(prediction_model_manifold_live.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

TARGET = "Output"

FEATURES = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
       'Educational Qualifications', 'Family size', 'latitude', 'longitude',
       'Pin code','Feedback']

