# import necessary libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
import random
from scipy.stats import rankdata
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.metrics import average_precision_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt


from Class import COVID_19_EWS
from functions import *

if __name__ == '__main__':
    
    # ==== data import ===
    # Validation data
    raw_val_data
    labels
    
    #Calibration data
    raw_cali_data
    labels_cali
    
    
    # import dependencies
    
    scaler = pickle.load(open('dependencies/scaler.sav', 'rb'))
    model = pickle.load(open('dependencies/model.sav', 'rb'))
    imputer_sex = pickle.load(open('dependencies/imputer_sex.sav', 'rb'))
    imputer_avpu = pickle.load(open('dependencies/imputer_avpu.sav', 'rb'))
    initial_imputer_sex = pickle.load(open('dependencies/initial_imputer_sex.sav', 'rb'))
    initial_imputer_avpu = pickle.load(open('dependencies/initial_imputer_avpu.sav', 'rb'))
    imputer = pickle.load(open('dependencies/imputer.sav', 'rb'))
    
    # Initiate class with imported data
    Class_init = COVID_19_EWS(scaler, model, imputer_sex, imputer_avpu,initial_imputer_sex,initial_imputer_avpu,imputer)
    Class_init.data_preprocessing(raw_val_data,raw_cali_data)
    Class_init.Model_updating(labels_cali)
    
    nboot = 500
    Class_init.model_validation(labels,nboot)