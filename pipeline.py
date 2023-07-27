#!/usr/bin/python 

import pandas as pd
import numpy as np
import pickle

from scripts.functions import run_pipeline, clean_prep

address = input("Pls give the address/location of the hold-out set: ")

#run the data_clean and prep function
data = clean_prep(df_address = address)

print('running pipeline...')

#run the pipeline to make predictions
score = run_pipeline(data)

print('The accuracy of the model is {} %'.format(round(score,2)))