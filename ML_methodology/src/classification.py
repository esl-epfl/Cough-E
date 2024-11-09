# Import libraries
from numpy import genfromtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import shap
import scipy.io as sio
import seaborn as sn

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit,GridSearchCV, cross_val_score, cross_val_predict, ShuffleSplit,learning_curve, train_test_split,LeaveOneOut
from sklearn.metrics import roc_curve, precision_recall_curve, plot_precision_recall_curve, auc,fbeta_score, make_scorer, recall_score, accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix, average_precision_score, f1_score, plot_confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import check_array

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from functools import partial

# to save the models
from joblib import dump, load 


# Return the optimal threshold that maximizes the F1 score
def get_opt_thresh(y_val,y_pred):
    """
    Inputs:
        - y_val: true labels
        - y_pred: predicte classifier probabilities
    Outputs:
        - best threshold maximizing F1 score
        - Maximum F-1 score
        - All true positive rates at every threshold
        - All false discovery rates at every threshold
    """
    tpr = []
    fdr = []
    f1s = []
    thr = np.arange(0,1,0.01)
    for thresh in thr:
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred[:,1]>=thresh).ravel()
        se = tp / (tp + fn) if (tp + fn)>0 else 0
        tpr.append(se)
        fd = fp / (fp + tp) if ((fp + tp))>0 else 0
        fdr.append(fd)
        ppv = tp / (fp + tp) if (fp + tp)>0 else 0
        f1 = 2*(ppv * se)/(ppv + se) if (ppv + se)>0 else 0
        f1s.append(f1)
    return thr[np.argmax(f1s)], np.max(f1s), tpr, fdr

#Custom pipeline object to use with RFECV
class PipelineRFE(imbpipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        model_type = type(self.steps[-1][-1])
        if model_type is SVC or model_type is LogisticRegression or model_type is LinearDiscriminantAnalysis:
            self.coef_ = self.steps[-1][-1].coef_
        elif model_type is XGBClassifier or model_type is DecisionTreeClassifier or model_type is RandomForestClassifier:
            self.feature_importances_ = self.steps[-1][-1].feature_importances_
        else:
            print("WARNING!! Model type is not compatible with RFECV")
        return self

def generate_pipeline(model_name, hps={}, smote=True, scaling=True):
    """
    Generate a pipeline using a model depending on the name of the model. The pipeline consists of:
    - the SMOTE algorithm to oversample the less dominant class
    - a standard scaler to standardize the features
    - a ML model
    
    Parameters:
    ----------------
    model_name: a one or two letter code indicating which model to use
        -LR = Logistic Regression
        -SVM = Support Vector Machines
        -KNN = K Nearest Neighbors
        -NB = Gaussian Naive Bayes
        -DTC = Decision Tree
        -RF = Random Forest
        -XGB = eXtreme Gradient Boosting
    hps: model hyperparameters
    smote: whether or not to oversample the less dominant class
    
    Returns:
    ----------------
    : the chosen pipeline
    """
    
    if model_name == 'LR':
        pipeline = PipelineRFE(steps = [['classifier', LogisticRegression(**hps,random_state=0)]])
    elif model_name == 'SVM':
        pipeline = PipelineRFE(steps = [['classifier', SVC(**hps,kernel='linear',random_state=0)]])
    elif model_name == 'KNN':
        pipeline = PipelineRFE(steps = [['classifier', KNeighborsClassifier(**hps,)]])
    elif model_name == 'NB':
        pipeline = PipelineRFE(steps = [['classifier', GaussianNB(**hps,)]])
    elif model_name == 'DTC':
        pipeline = PipelineRFE(steps = [['classifier', DecisionTreeClassifier(**hps,random_state=0)]])
    elif model_name == 'RF':
        pipeline = PipelineRFE(steps = [['classifier',RandomForestClassifier(**hps,random_state=0)]])
    elif model_name == 'XGB':
        pipeline = PipelineRFE(steps = [['classifier', XGBClassifier(**hps,seed=0,use_label_encoder=False, eval_metric='error')]]) 
    elif model_name == 'LDA':
        pipeline = PipelineRFE(steps = [['classifier', LinearDiscriminantAnalysis(solver='svd')]])
    else:
        print("ERROR: Invalid model name")
        pipeline = None
        
    #Add scaling
    if scaling:
        pipeline.steps.insert(0,['scaler', StandardScaler()])
    
    #Add SMOTE before scaling
    if smote:
        pipeline.steps.insert(0,['smote', SMOTE(random_state=0)])
    
    return pipeline

def train_model_untuned(X_train, y_train, subjects_train, cv, model_name, scoring_method, smote=True, scaling=True):
    """
    Train a given model without tuning hyperparameters; usually to determine which one performs the best.
    Performance is based on cross-validation accuracy.
    
    Parameters:
    ----------------
    X_train : feature matrix
    y_train : target array
    subjects_train: array of subject indices to ensure that subjects don't end up in both train and test in a CV split
    model_name: string indicating what type of model you want to optimize
    scoring_method: scoring method for CV such as roc_auc or f1_score
    smote: whether or not to oversample the less dominant class
    
    Returns:
    ----------------
    : mean CV score on desired scoring method
    : std CV score on desired scoring method
    : the trained model object
    """
    pipeline = generate_pipeline(model_name,smote=smote, scaling=scaling)
        
    score = cross_val_score(pipeline, X_train, y_train, groups=subjects_train, cv=cv, scoring=scoring_method)
    
    return score.mean(), score.std(), pipeline

def getSubjNamesFromFile(file_names):
    """Output an array of the subject numbers that each cough belongs to
    Input: List of file names where the file format name is fileName_subjectIndex.csv
    Output: np array of subject numbers to use for group shuffle split"""
    filenames_vec = []
    unique_filenames = []
    fns = file_names.copy()
    for fn in fns:
        fn_split = fn.split('_')
        fn_split.pop(-1)
        filenames_vec.append(fn_split)
    [unique_filenames.append(x) for x in filenames_vec if x not in unique_filenames]
    print(unique_filenames)
    subjVec = [unique_filenames.index(name) for name in filenames_vec]
    return np.asarray(subjVec)

# ------------------------------------------------- #
def cleaning(X, scaler,model_filename = "cough_detection_model.sav", threshold = 0.5):
    """
    Output a mask vector on the feature matrix 'X' based on the model 
    """
    # apply transform
    X = scaler.transform(X)
    cough_model = load("cough_detection_model.sav")
    y_cough = (cough_model.predict_proba(X)[:,1] >=threshold).astype(bool)
    return y_cough 

# ------------------------------------------------- #
#Define optimizer function to minimize for a given model and set of hyperparameters
def optimizer(hps, X_train, y_train, subjects_train, cv, model_name, scoring_method):
    """
    Target function for optimization to be minimized in hyperparameter selection
    Defined by the negative classification accuracy. Can be changed to include
    other metrics (ex. g-mean, computational complexity, energy consumption, etc)
    
    Parameters:
    ----------------
    hps : sample point from search space
    X_train : feature matrix
    y_train : target array
    subjects_train: array of subject ID numbers corresponding to each sample
    cv: cross-validation object
    model_name: string indicating what type of model you want to optimize
    scoring_method: scoring function of CV to optimize ex. roc_auc
    
    Returns:
    ----------------
    : target function value (negative CV accuracy)
    """
    pipeline = generate_pipeline(model_name, hps)
        
    score = cross_val_score(pipeline, X_train, y_train, groups=subjects_train, cv=cv, scoring=scoring_method)
    
    return -score.mean()

# ------------------------------------------------- #
#Loop through all models optimizing hyperparameters
def tuneHyperparameters(X_train, y_train, subjects_train, cv, model_names, scoring_method):
    """
    Tune the hyperparameters of a list of models and determine the one with the highest CV accuracy
    
    Parameters:
    ----------------
    hps : sample point from search space
    X_train : feature matrix
    y_train : target array
    subjects_train: array of subject ID numbers corresponding to each sample
    cv: cross-validation object
    model_name: string indicating what type of model you want to optimize
    scoring_method: scoring function of CV to optimize ex. roc_auc
    
    Returns:
    ----------------
    : accuracies of each tuned model
    : hyperparameter dictionaries of each model
    """
    model_results = []
    model_hyps = []
    for model_name in model_names:
        print(model_name)
        if model_name == 'LR':
            hp_space = {'C': hp.loguniform('C', -2*np.log(10.0), 6.0*np.log(10.0)), 
                        'class_weight': hp.choice('class_weight', [None, 'balanced']),
                        'solver': hp.choice('solver', ['newton-cg', 'lbfgs'])}
        elif model_name == 'SVM':
            hp_space = {'C': hp.loguniform('C', -2*np.log(10.0), 6.0*np.log(10.0))}
        elif model_name == 'KNN':
            hp_space = {'n_neighbors': hp.choice('max_depth',np.arange(1, round(X_train.shape[1]/2), dtype=int)), 
                        'weights': hp.choice('weights', ['uniform', 'distance'])}
        elif model_name == 'NB':
            hp_space = {'var_smoothing': hp.loguniform('var_smoothing', -10*np.log(10.0), 2*np.log(10.0))}
        elif model_name == 'DTC':
            hp_space = {'min_samples_split': hp.choice('min_samples_split',np.arange(2, 5, dtype=int)), 
                        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, X_train.shape[1], dtype=int)),
                        'max_features': hp.choice('max_features', np.arange(1, X_train.shape[1], dtype=int))} 
        elif model_name == 'RF':
            hp_space = {'n_estimators': hp.choice('n_estimators',np.arange(10, 300, dtype=int)), 
                        'max_features': hp.choice('max_features', np.arange(1, X_train.shape[1], dtype=int)),
                        'criterion': hp.choice('criterion', ['gini', 'entropy'])}
        elif model_name == 'XGB':
            hp_space = {'max_depth': hp.choice('max_depth',np.arange(1, 10, dtype=int)), 
                        'max_delta_step': hp.choice('max_delta_step',np.arange(1, 10, dtype=int)), 
                        'gamma': hp.uniform('gamma',0,1),
                        'subsample': hp.uniform('subsample',0,1),
                        'reg_lambda': hp.uniform('reg_lambda',0.5,2),
                        'eta': hp.uniform('eta',0,1),
                        'sampling_method': hp.choice('sampling_method', ['uniform', 'gradient_based'])}
            
        if model_name != 'LDA':
            trials_clf = Trials()
            best_clf = fmin(fn = partial(optimizer, X_train=X_train, y_train=y_train, subjects_train=subjects_train, cv=cv, model_name=model_name, scoring_method=scoring_method), 
                             space=hp_space, algo=tpe.suggest, max_evals=100, 
                             trials=trials_clf, rstate=np.random.RandomState(1))
            bayes_trials_results = sorted(trials_clf.results, key = lambda x: x['loss'])
            best_result = -bayes_trials_results[0]['loss']
            hyp = space_eval(hp_space, best_clf)
            model_results.append(best_result)
            model_hyps.append(hyp)
        else:
            lda = LinearDiscriminantAnalysis(solver='svd')
            scoresLDA = cross_val_score(lda, X_train, y_train, groups=subjects_train, cv=cv, scoring=scoring_method)
            best_result_LDA = scoresLDA.mean()
            model_results.append(best_result_LDA)
    
    return model_results, model_hyps

def plot_learning_curves(model_hyps, model_names, X, y, groups, axes=None, ylim=None, cv=None, scoring="roc_auc",
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate the test and training learning curve

    """

    if (len(model_names)>1):
        if axes is None:
            _, axes = plt.subplots(len(model_names), 1, figsize=(10, 40))
        for n in range(len(model_names)):
            title = model_names[n] + " Learning Curve"
            axes[n].set_title(title)
            if ylim is not None:
                axes[n].set_ylim(*ylim)
            axes[n].set_xlabel("Training examples")
            axes[n].set_ylabel("Score")
            
            if model_names[n] != "LDA":
                model = generate_pipeline(model_names[n], model_hyps[n])
            else:
                model = generate_pipeline(model_names[n])

            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(model, X, y, cv=cv, n_jobs=n_jobs,
                               train_sizes=train_sizes, scoring=scoring,
                               return_times=True, groups=groups, shuffle=True)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # Plot learning curve
            axes[n].grid()
            axes[n].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
            axes[n].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1,
                                 color="g")
            axes[n].plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
            axes[n].plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
            axes[n].legend(loc="best")
    else:
        plt.plot(figsize=(10,13))
        title = model_names[0] + " Learning Curve"
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        if model_names[0] != "LDA":
                model = generate_pipeline(model_names[0], model_hyps[0])
        else:
            model = generate_pipeline(model_names[0])
        
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(model, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, scoring=scoring,
                           return_times=True, groups=groups)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot learning curve
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        plt.legend(loc="best")


    return plt

def plot_auc(Yref, Yprob, thresh=0.5):
    """Get the F-1 score, confusion matrix, and ROC curve for a given classifier
    Yref: True output labels
    Yprob: Classifier output probabilities"""
    Ypred = Yprob > thresh
    f1_before_rfecv = f1_score(Yref, Ypred)
    print(f'Final score: {f1_score(Yref, Ypred):.8f}')
    class_report_before_rfecv = classification_report(Yref, Ypred, target_names=['Not Cough', 'Cough'])
    print(class_report_before_rfecv)
    print('Confusion Matrix')
    print(confusion_matrix(Yref, Ypred))
    roc_before_rfecv = roc_auc_score(Yref, Yprob)
    print(f'ROC AUC: {roc_before_rfecv:.8f}')

    #ROC Curve
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    fpr, tpr, thr = roc_curve(Yref, Yprob)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_before_rfecv)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title('ROC Curve')
    plt.legend()
    
def aggregate_results(subjects_test, y_prob, y_test, agg_method="logit_mean"):
    """Use the probabilities given for each cough to combine all coughs of a given subject into one probability
    inputs:
        subjects_test: vector of subject IDs corresponding to each prediction
        y_prob: output probabilities of each cough from classifier
        y_test: reference labels for each cough
    outputs:
        y_prob_aggregate: probability for each subject
        y_true_aggregate: label for each subject
        """
    ##Aggregated results
    subjects_test_unique = np.unique(subjects_test)
    print("There are {0} unique subjects".format(len(subjects_test_unique)))
    y_pred_all = []
    y_prob_aggregate = np.zeros(len(subjects_test_unique))
    y_true_aggregate = np.zeros(len(subjects_test_unique))
    for i,subj in enumerate(subjects_test_unique):
        pred_segment = y_prob[subjects_test==subj]
        y_pred_all.append(pred_segment)
        if agg_method == "mean":
            y_prob_aggregate[i] = np.mean([pred_segment])
        elif agg_method == "median":
            y_prob_aggregate[i] = np.median([pred_segment])
        elif agg_method == "max":
            y_prob_aggregate[i] = np.max([pred_segment])
        elif agg_method == "logit_prod":
            safety = 0.0001
            pred_segment[pred_segment == 0] = safety
            pred_segment[pred_segment == 1] = 1-safety
            y_prob_aggregate[i] = np.log(np.prod(pred_segment)/(np.prod(1-pred_segment)))
        elif agg_method == "logit_mean":
            safety = 0.0001
            pred_segment[pred_segment == 0] = safety
            pred_segment[pred_segment == 1] = 1-safety
            y_prob_aggregate[i] = np.mean(np.log(pred_segment/(1-pred_segment)))
        elif agg_method == "logit_median":
            safety = 0.0001
            pred_segment[pred_segment == 0] = safety
            pred_segment[pred_segment == 1] = 1-safety
            y_prob_aggregate[i] = np.median(np.log(pred_segment/(1-pred_segment)))
        elif agg_method == "logit_max":
            safety = 0.0001
            pred_segment[pred_segment == 0] = safety
            pred_segment[pred_segment == 1] = 1-safety
            y_prob_aggregate[i] = np.max(np.log(pred_segment/(1-pred_segment)))
        else:
            print("Error: Invalid aggregation method")
        y_true_aggregate[i] = np.all(y_test[subjects_test==subj])
        
    return y_prob_aggregate, y_true_aggregate

# ------------------------------------------------- #
def results_score(X_train, y_train, X_test, y_test, model, threshold = 0.5):
    y_pred_train = (model.predict_proba(X_train)[:,1] >=threshold).astype(bool)
   
    y_pred_test = (model.predict_proba(X_test)[:,1] >=threshold).astype(bool)
     # F1 score
    F1 = f1_score(y_train, y_pred_train, average='micro') 
    print('F1 score(micro) of classifier on train set: {:.4f}'
         .format(F1))
    
    F1 = f1_score(y_test, y_pred_test, average='micro') 
    print('F1 score(micro) of classifier on test set: {:.4f}'
         .format(F1))
    
    F1 = f1_score(y_train, y_pred_train, average='macro') 
    print('F1 score(macro) of classifier on train set: {:.4f}'
         .format(F1))
          
    F1 = f1_score(y_test, y_pred_test, average='macro') 
    print('F1 score(macro) of classifier on test set: {:.4f}'
         .format(F1))

    # confusion matrix
    y_test = y_test.astype(int)
    print('y_test')
    print(y_test)
    y_pred_test = y_pred_test.astype(int)
    print('y_pred')
    print(y_pred_test)
    TN, FP, FN, TP = confusion_matrix(y_test,y_pred_test).ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('TPR(sensivity) of classifier on test set: {:.4f}'.format(TPR))
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print('FPR(Fall out/False positive rate) of classifier on test set: {:.4f}'.format(FPR))
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print('TNR(specificity) of classifier on test set: {:.4f}'.format(TNR))
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('PPV(precision) of classifier on test set: {:.4f}'.format(PPV))
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('NPV of classifier on test set: {:.4f}'.format(PPV))
    # False negative rate
    FNR = FN/(TP+FN)
    print('FNR of classifier on test set: {:.4f}'.format(FNR))
    # False discovery rate
    FDR = FP/(TP+FP)
    print('FDR(false discovery rate) of classifier on test set: {:.4f}'.format(FDR))

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Accuracy of classifier on test set: {:.4f}'.format(ACC))
    
    
    