class COVID_19_EWS(object):
     
    def __init__(self, scaler, model, imputer_sex, imputer_avpu,initial_imputer_sex,initial_imputer_avpu,imputer):
        
        self.scaler = scaler
        self.model = model
        self.imputer_sex = imputer_sex
        self.imputer_avpu = imputer_avpu
        self.initial_imputer_sex = initial_imputer_sex
        self.initial_imputer_avpu = initial_imputer_avpu
        self.imputer = imputer
        self.cols = ['Sex','Age','Length-of-stay','O2','O2','SpO2/O2','SpO2','HR','SBP','RR','Temp','AVPU','SpO2_diff','HR_diff',
                'SBP','RR_diff','Temp_diff','SpO2/O2_diff']
    
    def data_preprocessing(self,raw_val_data,raw_cali_data):
        
        X_norm = normalize(raw_val_data,self.cols,self.scaler) # normalize data first (except categorical variables)
        X_norm = impute_categorical(X_norm,'Sex',self.imputer_sex,self.initial_imputer_sex) # impute missing Sex
        X_norm = impute_categorical(X_norm,'AVPU',self.imputer_avpu,self.initial_imputer_avpu) # impute missing AVPU
        self.X_norm_imp_val = imputer(X_norm,self.imputer) 
        
        X_norm = normalize(raw_cali_data,self.cols,self.scaler) # normalize data first (except categorical variables)
        X_norm = impute_categorical(X_norm,'Sex',self.imputer_sex,self.initial_imputer_sex) # impute missing Sex
        X_norm = impute_categorical(X_norm,'AVPU',self.imputer_avpu,self.initial_imputer_avpu) # impute missing AVPU
        self.X_norm_imp_cali = imputer(X_norm,self.imputer) 
    
        
    def Model_updating(self,labels_cali):
        
        self.updated_model = calibrate(self.model,self.X_norm_imp_cali,labels_cali)
        
    def model_validation(self,labels,nboot):
        
        predictions = self.updated_model.predict_proba(self.X_norm_imp_val)[:,1] # make predictions
        
        ci = p_Boot(labels,predictions,nboot,t=0.33, partial=True)
        print('pAUC:' + str(ci[0]) + '[' + str(ci[1][0]) + ' - ' + str(ci[1][1]) + ']')
        
        ci = p_Boot(labels,predictions,nboot,t=0.33)
        print('AUC:' + str(ci[0]) + '[' + str(ci[1][0]) + ' - ' + str(ci[1][1]) + ']')
        
        ci = p_Boot(labels,predictions,nboot,t=0.33,AUCPR=True)
        print('AUCPR:' + str(ci[0]) + '[' + str(ci[1][0]) + ' - ' + str(ci[1][1]) + ']')
        
        mean_predicted_value,y,fracs_lower,fracs_upper = prepare_calibration_plot(labels,predictions)
        plot_calibration_curve(predictions,labels,mean_predicted_value,y,fracs_lower,fracs_upper)
        