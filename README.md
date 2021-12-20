# COVID-19 EWS
A new early warning model developed specifically for hospitalized COVID-19 patients.

## Brief Introduction
Timely identification of deteriorating COVID-19 patients is needed to guide changes in clinical management and admission to intensive care units (ICUs). There is widespread concern that widely used early warning scores (EWSs), like the Modified Early Warning Score (MEWS), National Early Warning
Score (NEWS) and its successor, NEWS2, underestimate illness severity in COVID-19 patients. Therefore, we developed a new early warning model for longitudinal monitoring of hospitalized COVID-19
patients. 

## Data
These instructions go through the loading of the fitted models, pre-processing of the raw data and evaluation of our model on your own dataset.

You need to upload 2 datasets, both a set used to fit a calibrator (calibration dataset) and a set used to validate the model (validation dataset). 
Both the datasets need to be imported in the main file, where the path where the data is saved needs to be specified. These datasets need to be [N x M] sized tables, where N (nuber of rows) depicts the individual patient samples and M (number of columns) the variables collected for each sample. Also the corresponding [N x 1] label vectors (0=negative, 1=positive) need to be loaded. 
Label the samples as positive if unplanned ICU admission or death occurred within 24 hours from the moment of sampling, and negative otherwise.

Furthermore, make sure the columns of the datasets are in the correct order and the values are in the correct units (see the table below).



column | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11| #12 | #13 | #14 | #15 | #16 | #17 | #18
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |--- |--- |--- |--- |---
Variable | Sex | Age | Length-of-stay | O2 | O2 | SpO2/O2 | SpO2 | Heart rate | Systolic blood pressure | Respiratory rate | Temperature| AVPU | ΔSpO2 | ΔHeart rate | ΔSystolic blood pressure | ΔRespiratory rate | ΔTemperature | ΔSpO2/O2
Unit | 0=female, 1=male | years | hours | 0=no, 1=yes | L/min | %/(L/min) | % | bpm | mmHg | /min | °C| - | % | bpm | mmHg | /min | °C | %/(L/min)


## Validate model
By running the main code, the 
- trained model presented in the study, as well as the fitted normalization and imputation functions are loaded from the 'dependencies' directory
- your uploaded datasets are preprocessed, i.e. normalized and imputed
- a calibrator is fitted using the predictions of the trained model to the calibration dataset and the corresponding labels
- predictions are made for the validation dataset using the resulting model-calibrator pair 
- These predictions are evaluated for discrimination and calibration

## Run
python main.py
