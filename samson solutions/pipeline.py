#!/usr/bin/python 

import pandas as pd
import numpy as np
import pickle

from functions import run_pipeline,data_clean_prep

address = input("Pls give the address/location of the hold-out set: ")

#run the data_clean and prep function
data = data_clean_prep(df_address = address)

print('running pipeline...')

#run the pipeline to make predictions
score = run_pipeline(data)

print('The accuracy of the model is {} %'.format(round(score,2)))