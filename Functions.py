# Author: Linda Karlsson, 2022

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD
import sklearn as skl
import statsmodels.formula.api as smf

"""
Returns the intersection of two lists.

Inputs:
    - lst1 (list): List 1.
    - lst2 (list): List 2.
    
Outputs: 
    - lst3 (list): Intersection of lst1 and lst2.
"""
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


"""
Returns linear regression coefficients and p-values after standardization of all variables. 

Inputs: 
    - df (DataFrame): DataFrame including main predictor, outcome and confounders.
    - main_pred (String): Name of main predictor.
    - outcome (String): Name of outcome.
    - confounders (List): List of names of confounders.
    
Outputs:
    - betas (Series): Series of beta coefficients.
    - pvals (Series): Series of p-values.

"""
def get_linreg(df,main_pred,outcome,confounders=[]):
    X = df[[main_pred,outcome] + confounders]
    X_n = StandardScaler().fit_transform(X)
    X_n = pd.DataFrame(X_n, columns=X.columns)
    preds = main_pred
    for con in confounders:
        preds = preds + '+' + con
    model = smf.ols(outcome + '~' + preds, data=X_n,missing='drop').fit(disp=False)
    betas = model.params
    pvals = model.pvalues
    return betas,pvals


"""
Loops over a list of protein candidates that can strenghten the association between the baseline variables and
y. Data from each candidate should exist in df. Returns the best candidates and their corresponding AUC scores.

Inputs: 
    - df (DataFrame): DataFrame with columns of independent variables/inputs/predictors to be fitted against.
    - y (list, DataFrame): dependent variable/output/response variable of logistic regression.
    - candidates (list): list of candidate names. Each candidate should have a column in df.
    - baseline_vars (list): name of baseline variables to be used as predictors in the regression model.
    - amount (int): amount of best candidates to return (defaults to 10).
    - n_splits (int): number of splits in cross-validation (defaults to 10).
    
Outputs: 
    - best_AUC (list): list of amount highest AUC scores (decreasing order).
    - names_best_AUC (list): list of corresponding best_AUC candidate names (decreasing order).
"""
def get_best_candidates(df,y_true,candidates,baseline_vars,amount=10,n_splits=10):
    best_AUC = np.zeros(amount)
    names_best_AUC = np.zeros(amount).astype(str)

    for candidate in tqdm(candidates):
        include = baseline_vars.copy()
        include.append(candidate)
        X = df[include]
        X = X.dropna()
        y = np.array(y_true)[X.index]
        
        mean_auc = get_mean_AUC_score(X,y,include,n_splits=n_splits)
        i = 0
        while (i < len(best_AUC) and ((mean_auc < best_AUC[i]) and ((best_AUC[i]) != 0))):
            i += 1
        if i <= len(best_AUC)-1:
            best_AUC = np.insert(best_AUC, i, mean_auc)
            best_AUC = np.delete(best_AUC,amount)
            names_best_AUC = np.insert(names_best_AUC,i,candidate)
            names_best_AUC = np.delete(names_best_AUC,amount)
            
    return best_AUC, names_best_AUC


"""
Computes the mean AUC score after performing a cross-validation.

Inputs:
    - X (DataFrame): DataFrame containing subjects and relevant variables.
    - y (array): Array of labels for corresponding subjects in X.
    - variables (list): List of columns in X to use in the classifier.
    - n_splits (int): Split in cross-validation (defaults to 10).
    - clf (classifier): model to train for classification (defaults to LogisticRegression()).
    - scoring (String): scoring to evaluate (defaults to 'roc_auc').

Outputs:
    - mean_score (int): mean score from cross-validation.
"""
def get_mean_AUC_score(X,y,variables,n_splits=10,clf = LogisticRegression(),scoring='roc_auc'):
    scores = cross_val_score(clf, X[variables], y, cv=n_splits,scoring=scoring)
    mean_score = np.mean(scores)
    return mean_score


"""
Splits X and y into folds, standard scales them (fitted from X_train), and prepares each X_test fold with selected noise type and noise level. 

Inputs:
    - X (DataFrame): Dataframe containing data for all subjects and candidates.
    - y (array): Array of labels for corresponding subjects in X.
    - noise (int): 1 (measurement noise) or 2 (assay drift).
    - candidates (list): List of protein candidates.
    - strength (float): Strength of noise/noise level (defaults to 0).
    - n_splits (int): number of folds to split data into (deafults to 10).
    
Outputs:
    - X_trains (list): List of n_splits standardized X_train sets.
    - X_tests (list): List of n_splits standardized X_test sets with added noise.
    - y_trains (list): List of n_splits y_train sets.
    - y_tests (list): List of n_splits y_test sets.
"""
def prep_folds_with_noise(X,y,nt,candidates, strength=0.0, n_splits=10):
    kf = KFold(n_splits = n_splits)
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler().fit(X_train)
        X_train_n = scaler.transform(X_train)
        
        X_test_n = add_noise(X_test, nt=nt, strength=strength,candidates=candidates)
        X_test_n = scaler.transform(X_test_n)

        X_trains.append(X_train_n)
        X_tests.append(X_test_n)
        y_trains.append(y_train)
        y_tests.append(y_test)
        
    return X_trains, X_tests, y_trains, y_tests


"""
Adds selected noise to data.

Inputs: 
    - X (array): Data to add noise to.
    - nt (int): 1 (measurement noise) or 2 (assay drift).
    - strength (float): Strength of noise/noise level.
    - candidates (list): List of protein candidates.
    
Outputs:
    - X_noise (array): Data with added noise.
     
"""
def add_noise(X,nt,strength,candidates):
    X_noise = X.copy()
    X_use = X_noise[candidates]
    if nt == 1:
        noise = strength * np.random.randn(X_use.shape[0],X_use.shape[1])
        X_noise[candidates] = X_use + noise*np.outer(np.std(X_use,axis=0),np.ones((1,X_use.shape[0]))).T
    elif nt == 2:
        noise = (np.random.randint(0,2,size=X_use.shape[1])*2-1)*strength
        X_noise[candidates] = X_use + noise*X_use
    else:
        X_noise = X
        
    return X_noise


"""
Trains and evaluates a logistic regression according to specific train/test-folds.

Inputs:
    - X_train (array): Train data.
    - X_test (array): Test data.
    - y_train (array): Train labels.
    - y_test (array): Test labels.
Outputs:
    - score (float): Corresponding AUC score for test data.

"""
def train_and_predict(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
    return score


"""
Trains and evaluates a logistic regression using either 1, a single reference candidate or 2, the SVD of several reference candidates.

Inputs:
    - X_train (array): Train data.
    - X_test (array): Test data.
    - y_train (array): Train labels.
    - y_test (array): Test labels.
Outputs:
    - score (float): Corresponding AUC score for test data.

"""
def get_svd_score(X_train,X_test,y_train,y_test,candidates,pred):
    if len(candidates) == 1:
        X_train['ref_svd'] = X_train[candidates]
        X_test['ref_svd'] = X_test[candidates]
    else:
        svd = TruncatedSVD(n_components=1, n_iter=5, random_state=0)
        svd.fit(X_train[candidates])
        X_train['ref_svd'] = svd.transform(X_train[candidates])
        X_test['ref_svd'] = svd.transform(X_test[candidates])
    
    X_train = X_train[pred + ['ref_svd']]
    X_test = X_test[pred + ['ref_svd']]
    score = train_and_predict(X_train, y_train, X_test, y_test)
    return score


"""
Bootstraps the roc auc scores for a selected classifier.

Inputs: 
    - X (DataFrame): Dataframe containing data for all subjects and candidates.
    - y (array): Array of labels for corresponding subjects in X.
    - kind (int): 0 for test with a single reference, 1 for test with svd of several references and 2 for test with mean level of several references (defaults to 0).
    - prots (list): List of reference proteins, only needed if kind = 1 or 2 (defaults to None).
    - n_iter (int): number of iterations in bootstrap (defaults to 1000).
    - clf (sklearn classifier): classifier used for prediction (defaults to LogisticRegression()).
    
Outputs: 
    - roc_auc_scores (list): list of the n_iter resulting roc auc scores.
"""
def bootstrap_roc_auc(X,y,kind = 0,prots = None,n_iter=2000,clf=LogisticRegression()):
    roc_auc_scores = []
    for n_iter in tqdm(range(n_iter)):
        clf_use = skl.base.clone(clf)
        
        idx_train = np.random.choice(len(X), size=len(X), replace=True)
        X_train = X.iloc[idx_train]
        y_train = y[idx_train]
        scaler = StandardScaler().fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_train_n = pd.DataFrame(X_train_n,columns=X_train.columns)
            
        idx_val = np.array([x not in idx_train for x in np.arange(len(X))])
        X_val = X.iloc[idx_val]
        y_val = y[idx_val]
        X_val_n = scaler.transform(X_val)
        X_val_n = pd.DataFrame(X_val_n,columns=X_val.columns)
        
        if kind == 1:
            svd = TruncatedSVD(n_components=1, n_iter=5, random_state=0)
            svd.fit(X_train_n[prots])
            X_train_n['ref_svd'] = svd.transform(X_train_n[prots])
            X_train_n = X_train_n.drop(columns=prots)
            
            X_val_n['ref_svd'] = svd.transform(X_val_n[prots])
            X_val_n = X_val_n.drop(columns=prots)
        
        if kind == 2:
            X_train_n['mean'] = np.mean(X_train_n[prots],axis=1)
            X_train_n = X_train_n.drop(columns=prots)
            
            X_val_n['mean'] = np.mean(X_val_n[prots],axis=1)
            X_val_n = X_val_n.drop(columns=prots)

        clf_use.fit(X_train_n,y_train)
        roc_auc_scores.append(roc_auc_score(y_val,clf_use.predict_proba(X_val_n)[:, 1]))
        
    return roc_auc_scores



"""
Bootstraps and compares roc auc scores using two references. Results can be used to test significance.

Inputs: 
    - X (DataFrame): Dataframe containing data for all subjects and candidates.
    - y (array): Array of labels for corresponding subjects in X.
    - kind1 (int): 0 for test with a single or no reference, 1 for test with svd of several references and 2 for test with mean level of several references (defaults to 0).
    - kind2 (int): 0 for test with a single or no reference, 1 for test with svd of several references and 2 for test with mean level of several references (defaults to 0).
    - main (list): List of main predictors.
    - prots1 (list): List of reference proteins for first reference.
    - prots2 (list): List of reference proteins for second reference.
    - n_iter (int): number of iterations in bootstrap (defaults to 1000).
    - clf (sklearn classifier): classifier used for prediction (defaults to LogisticRegression()).
    
Outputs: 
    - roc_auc_diffs (list): list of to n_iter roc auc differences between the two models.
    - aucs1 (list): list of the n_iter resulting roc auc scores for model with first reference.
    - aucs2 (list): list of the n_iter resulting roc auc scores for model with second reference.
"""
def test_bootstrap_roc_auc(X,y, kind1 = 0, kind2 = 0, main =[], prots1 = [], prots2 = [],n_iter=2000,clf=LogisticRegression()):
    roc_auc_diffs = []
    aucs1 = []
    aucs2 = []
    for n_iter in tqdm(range(n_iter)):
        clf_use1 = skl.base.clone(clf)
        clf_use2 = skl.base.clone(clf)
        idx_train = np.random.choice(len(X), size=len(X), replace=True)
        X_train = X.iloc[idx_train]
        y_train = y[idx_train]
        scaler = StandardScaler().fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_train_n = pd.DataFrame(X_train_n,columns=X_train.columns)


        idx_val = np.array([x not in idx_train for x in np.arange(len(X))])
        X_val = X.iloc[idx_val]
        y_val = y[idx_val]
        X_val_n = scaler.transform(X_val)
        X_val_n = pd.DataFrame(X_val_n,columns=X_val.columns)

        ref1 = prots1
        ref2 = prots2

        if kind1 == 1:
            svd = TruncatedSVD(n_components=1, n_iter=5)
            svd.fit(X_train_n[prots1])
            X_train_n['ref1'] = svd.transform(X_train_n[prots1])
            X_val_n['ref1'] = svd.transform(X_val_n[prots1])
            ref1 = ['ref1']
        elif kind1 == 2:
            X_train_n['ref1'] = np.mean(X_train_n[prots1],axis=1)
            X_val_n['ref1'] = np.mean(X_val_n[prots1],axis=1)
            ref1 = ['ref1']

        if kind2 == 1:
            svd = TruncatedSVD(n_components=1, n_iter=5)
            svd.fit(X_train_n[prots2])
            X_train_n['ref2'] = svd.transform(X_train_n[prots2])
            X_val_n['ref2'] = svd.transform(X_val_n[prots2])
            ref2 = ['ref2']
        elif kind1 == 2:
            X_train_n['ref2'] = np.mean(X_train_n[prots2],axis=1)
            X_val_n['ref2'] = np.mean(X_val_n[prots2],axis=1)
            ref2 = ['ref2']

        clf_use1.fit(X_train_n[main + ref1],y_train)
        clf_use2.fit(X_train_n[main + ref2],y_train)         
        auc1 = roc_auc_score(y_val,clf_use1.predict_proba(X_val_n[main + ref1])[:, 1])
        auc2 = roc_auc_score(y_val,clf_use2.predict_proba(X_val_n[main + ref2])[:, 1])               
        roc_auc_diffs.append(auc2-auc1)
        aucs1.append(auc1)
        aucs2.append(auc2)
    
    return roc_auc_diffs,aucs1,aucs2
