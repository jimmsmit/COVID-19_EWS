def normalize(X,cols,scaler):
    """""
    Returns normalized data.

        Parameters:
                X (pandas DataFrame): the raw data
                cols (list): column names
                scaler (object): Fitted scaler as returned by scikit learn preprocessing.StandardScaler

        Returns:
                X_norm (pandas DataFrame): normalized data
    """""
    
    X = pd.DataFrame(raw_table)
    X.columns = cols
    sex = X.Sex
    avpu = X.AVPU
    X_norm = pd.DataFrame(scaler.transform(X))
    X_norm.loc[:,'Sex'] = sex
    X_norm.loc[:,'AVPU'] = avpu
    
    return X_norm

def impute_categorical(X,name,imputer,initial_imputer):
    """""
    Returns data with imputed values for missing values of categorical variables

        Parameters:
                X (pandas DataFrame): the normalized, non-imputed data
                name (string): name of column to be imputed
                imputer (object): Fitted model to impute categorical variable, as returned by scikit learn linear_model.LogisticRegression
                initial_imputer (object): Fitted imputer to initially impute the remaining variables in the temporal dataset, as returned by scikit learn impute.SimpleImputer

        Returns:
                X (pandas DataFrame): data with imputed values for missing values of categorical variables
    """""
    
    X_temp = X.drop([name], axis=1) #make copy of X_train without label column
    X_temp = initial_imputer.transform(X_temp) # impute remaining missing values values
        
    idx = X.index[X[name].apply(np.isnan)] # find idx at X_train where sex is missing
    for i in idx:
        X.loc[i,name] = imputer.predict(X_temp[i,:].reshape(1,-1)) 
    
    return X

def imputer(X,imputer):
    """""
    Returns data with imputed values for missing values of continuous variables

        Parameters:
                X (pandas DataFrame): the normalized, non-imputed data
                imputer (object): Fitted imputer to impute continous variables, as returned by scikit learn impute.IterativeImputer
                

        Returns:
                X_norm_imp (pandas DataFrame): data with imputed values for missing values of continuous variables
    """""
    
    X_norm_imp = imputer.transform(X)
    return X_norm_imp


def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def calibrate(model,X_cali,labels_cali):
    """""
    Returs model-calibrator pair to make calibrated predictions.

        Parameters:
                model (object): Fitted classification model without updating. 
                X_cali (pandas DataFrame): part of the new dataset to be used to fit the calibrator
                labels_cali (numpy array): corresponding labels of part of the new dataset to be used to fit the calibrator

        Returns:
                calibrated_model (object): classifier and calibrator pair as returned by scikit learn calibration.CalibratedClassifierCV
    """""
    
    calibrated_model = CalibratedClassifierCV(base_estimator=model,method='isotonic',cv='prefit')
    calibrated_model.fit(X_cali, labels_cali)
    
    return calibrated_model

def p_Boot(labels,predictions,nboot,t=0, partial=False,AUCPR=False):
    """""
    Returns the point estimate and 95% confidence intervals for the predictive performance metrics AUC, partial AUC and AUCPR

        Parameters:
                labels (numpy array): ground-truth labels corresponding to the part of the new dataset to be used to validate the model
                predictions (numpy array): predictions made by the model-calibrator pair for the part of the new dataset to be used to validate the model
                nboot (int): number of bootstrap samples to be taken to assess the bootstrap 95% confidence intervals
                t (float): threshold of the false-positive-rate to be used to calculated the partial AUC
                AUCPR (boolean): if True, calculate the AUCPR, if False, calculate (partial) AUC
                partial (boolean): if True, calculate the partial AUC, if False, calculate full AUC. Ignored if AUCPR = True
                
        Returns:
                A (float): Point estimate of the calculated performance metric
                Aci (list): lower and upper bound of the 95% confidence interval of the calculated performance metric
    """""
    
    
    df = pd.DataFrame()
    df['label'] = labels
    df['pred'] = pred
    
    if AUCPR:
        A = average_precision_score(labels, predpredictions
    else:
        if partial:
            A = roc_auc_score(labels,predictions,max_fpr=t)
        else:
            A = roc_auc_score(labels,predictions)
       
    n = sum(labels) # number of positives
    m = len(labels) - n  # number of negatives
    N = int(m+n)

    A_boot = []
    for i in range(nboot):
        pred_boot, labels_boot = resample(predictions, labels, stratify=labels, random_state=i)
        if AUCPR:
            A_boot.append(average_precision_score(labels_boot, pred_boot))
        else:
            if partial:
                A_boot.append(roc_auc_score(labels_boot, pred_boot,max_fpr=t))
            else:
                A_boot.append(roc_auc_score(labels_boot, pred_boot))
        
    Aci = [np.percentile(A_boot, 2.5, axis=0),np.percentile(A_boot, 97.5, axis=0)]
    
    
    return A, Aci

def Validation_slope_intercept(preds,labels):
    """""
    Returns the point estimate and 95% confidence intervals for the calibration intercept and slope

        Parameters:
                preds (numpy array): predictions made by the model-calibrator pair for the part of the new dataset to be used to validate the model
                labels (numpy array):  ground-truth labels corresponding to the part of the new dataset to be used to validate the model

        Returns:
                inter (float): point estimate of the calibration intercept
                inter_low (float): lower bound of 95% CI of the calibration intercept
                inter_high (float): upper bound of 95% CI of the calibration intercept
                slope (float): point estimate of the calibration slope
                slope_low (float): lower bound of 95% CI of the calibration slope
                slope_high (float): upper bound of 95% CI of the calibration slope
                
    """""
    preds[preds==0] = 0.0001
    preds[preds==1] = 0.9999
 
    df = pd.DataFrame()
    df['y'] = labels
    df['logitp'] = logit(preds)
    
    # fit logit model for slope
    formula = 'y ~ logitp'  
    model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
    result = model.fit()
    slope = result.params[1] 
    slope_low = result.conf_int(alpha=0.05)[0][1]
    slope_high = result.conf_int(alpha=0.05)[1][1]
    
    formula = 'y ~ 1'
    model = smf.glm(formula = formula, data=df, family=sm.families.Binomial(),offset=df['logitp'])
    result = model.fit()

    inter = result.params[0] 
    inter_low = result.conf_int(alpha=0.05)[0][0]
    inter_high = result.conf_int(alpha=0.05)[1][0]
    
    return inter, inter_low, inter_high, slope, slope_low, slope_high


def lowess_bell_shape_kern(x, y, tau = .005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve. 
    """
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest

def prepare_calibration_plot(labels,predictions):
    """
    Returns data needed to plot the calibration curve. 
    
        Parameters:
                labels (numpy array):  ground-truth labels corresponding to the part of the new dataset to be used to validate the model
                predictions (numpy array): predictions made by the model for the part of the new dataset to be used to validate the model
                
        Returns:
                mean_predicted_value (numpy array): The mean predicted probability in each bin, as returned by .calibration.calibration_curve
                y (numpy array): smoothed fractions of positives in each bin
                fracs_lower (numpy array): Lower bound of the 95% CI around smoothed fractions of positives in each bin
                fracs_upper (numpy array): Upper bound of the 95% CI around smoothed fractions of positives in each bin
    
    """
    strategy='uniform'
    mask = predictions<=0.2
    perc = sum(mask)/len(predictions)*100 
    print(perc, ' % of predictions in 0 - 0.2 probability range')
    
    fracs = []
    mean = np.linspace(0, 1, 100)
    B = nboot
    n_bins = 40
    
    for b in range(B):
        pred_boot, labels_boot = resample(predictions, labels, stratify=labels, random_state=b)
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(labels_boot, pred_boot, n_bins=n_bins,strategy=strategy)
        try:
            y = lowess_bell_shape_kern(mean_predicted_value, fraction_of_positives)
            interp_frac = np.interp(mean, mean_predicted_value, y)
        except:
            print('singular matrix')

        fracs.append(interp_frac)
        
    fracs_lower = np.percentile(fracs,2.5,axis=0)
    fracs_upper = np.percentile(fracs, 97.5,axis=0)

    return mean_predicted_value,y,fracs_lower,fracs_upper



    fraction_of_positives, mean_predicted_value = \
            calibration_curve(labels, predictions, n_bins=n_bins,strategy=strategy)
    y = lowess_bell_shape_kern(mean_predicted_value, fraction_of_positives)
    
    
    return mean_predicted_value,y,fracs_lower,fracs_upper
    

def plot_calibration_curve(predictions,labels,mean_predicted_value,y,fracs_lower,fracs_upper):
    """
    Creates plot of calibration curve including calibration intercept and slope.

        Parameters:
                labels (numpy array): ground-truth labels corresponding to the part of the new dataset to be used to validate the model
                predictions (numpy array): predictions made by the model-calibrator pair for the part of the new dataset to be used to validate the model
                mean_predicted_value (numpy array): The mean predicted probability in each bin, as returned by .calibration.calibration_curve
                y (numpy array): smoothed fractions of positives in each bin
                fracs_lower (numpy array): Lower bound of the 95% CI around smoothed fractions of positives in each bin
                fracs_upper (numpy array): Upper bound of the 95% CI around smoothed fractions of positives in each bin
       
    """

    inter, inter_low, inter_high, slope, slope_low, slope_high = Validation_slope_intercept(predictions,labels)
    
    
    plt.figure(figsize=(27, 20))
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 4), (2, 0),colspan=4)
    ax3.set_yscale("log")
    
    # perfect
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",linewidth=3)
    ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",linewidth=3)
    
    
    ax1.plot(mean_predicted_value, y, 
               label='Smoothed calibration curve: Intercept: '+ str(np.round(inter,2)) + ' (' + str(np.round(inter_low,2)) + ';' + str(np.round(inter_high,2)) + ')' + ', Slope:' + str(np.round(slope,2)) + ' (' + str(np.round(slope_low,2))+ ';' + str(np.round(slope_high,2)) + ')',
              )
    ax2.plot(mean_predicted_value, y, 
              label= 'Smoothed calibration curve (zoom-in)')
    
    ax1.fill_between(mean, fracs_lower, fracs_upper, alpha=.2,
                     where=(mean>=np.min(mean_predicted_value))&(mean<=np.max(mean_predicted_value)))
    ax2.fill_between(mean, fracs_lower, fracs_upper, alpha=.2,
                      where=(mean>=np.min(mean_predicted_value))&(mean<=np.max(mean_predicted_value)))
    
    ax3.hist(predictions, range=(0, 1), bins=40,
             label = str(np.round(perc,1)) + '% of predictions within 0-0.2 range (grey area).',
              histtype="step", lw=2)
    
    ax1.set_ylabel("Observed proportion")
    ax1.set_ylim([-0.01, 1.05])
    ax1.set_xlim([-0.01, 1.05])
    ax1.set_xlabel("Predicted probability",fontsize=20)
    
    ax2.set_ylim([0, 0.22])
    ax2.set_xlim([0, 0.22])
    ax2.set_xlabel("Predicted probability",fontsize=20)
    
    ax1.legend(loc="upper left",prop={'size':18})
    ax2.legend(loc="upper left",prop={'size':18})
        
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)
    ax3.set_xlabel("Predicted probability",fontsize=25)
    ax1.set_ylabel("Observed proportion",fontsize=25)
    ax3.set_ylabel("Count",fontsize=25)
    ax3.legend(loc="upper center",  prop={'size':18})
    
    ax1.axvspan(0,0.2,ymax=0.2, facecolor='grey',alpha=0.09)
    ax2.axvspan(0,0.2,ymax=(10/11), facecolor='grey',alpha=0.09)
    ax3.axvspan(0,0.2,facecolor='grey',alpha=0.09)
    
    ax2.set_xticks([0.0,0.05,0.1,0.15,0.2])
    ax2.set_yticks([0.0,0.05,0.1,0.15,0.2])
    
    
    plt.show()
