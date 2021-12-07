# COVID-19 EWS
A new early warning model developed specifically for hospitalized COVID-19 patients.

## Brief Introduction
Timely identification of deteriorating COVID-19 patients is needed to guide changes in clinical management and admission to intensive care units (ICUs). There is widespread concern that widely used early warning scores (EWSs), like the Modified Early Warning Score (MEWS), National Early Warning
Score (NEWS) and its successor, NEWS2, underestimate illness severity in COVID-19 patients. Therefore, we developed a new early warning model for longitudinal monitoring of hospitalized COVID-19
patients. 

## Data
These instructions go through the loading of the fitted models, pre-processing of the raw data and evaluation of our model on your own dataset.

To download and build the datasets run:


column | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11| #12 | #13 | #14 | #15 | #16 | #17 | #18
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |--- |--- |--- |--- |---
Variable | Sex | Age | Length-of-stay | O2 | O2 | SpO2/O2 | SpO2 | Heart rate | Systolic blood pressure | Respiratory rate | Temperature| AVPU | ΔSpO2 | ΔHeart rate | ΔSystolic blood pressure | ΔRespiratory rate | ΔTemperature | ΔSpO2/O2
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |--- |--- |--- |--- |---
Unit | 0=female, 1=male | years | hours | 0=no, 1=yes | L/min | %/(L/min) | % | bpm | mmHg | /min | °C| - | % | bpm | mmHg | /min | °C | %/(L/min)
