import pandas as pd
import numpy as np 
from prediction_model_manifold_live.config import config
from prediction_model_manifold_live.preprocessing import preprocess as pp
from prediction_model_manifold_live.preprocessing import data_handling
import prediction_model_manifold_live.pipeline as pipe
import sys

def perform_training():
    train_data = data_handling.load_dataset(config.TRAIN_FILE) # create load_dataset, create train_file in config
    train_target = train_data[config.TARGET].map({"Yes":1,"No":0}) # config - TARGET
    pipe.classification_model.fit(train_data[config.FEATURES], train_target) # create model pipleine in pipeline module, configure fratures in config
    data_handling.save_pipeline(pipe.classification_model)

if __name__=="__main__":
    perform_training()


