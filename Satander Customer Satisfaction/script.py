# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:54:05 2016

@author: Thiru
"""

##FURTHER CODE CLEANING IS NEEDED

import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib

#main preprocessing function.
##Inputs: 
#1. <Pandas DataFrame> df: training data
#2. <Pandas DataFrame> df2: testing data
#3. <float> correlationThreshold: remove features with correlation greater than this value
#4. <int> cutTillNum: remove features which are correlated to at least this value of columns
##Outputs:
#1. preprocessed training data
#2. preprocessed testing data
def preProcess(df,df2,correlationThreshold,cutTillNum):
    ##remove constant features  
    def removeConstantsTrainTest(train,test):
        constantColumns=[]
        for col in train.columns:
            if train[col].std() == 0:
                constantColumns.append(col)
        
        train.drop(constantColumns, axis=1, inplace=True)
        test.drop(constantColumns, axis=1, inplace=True)
        return train,test
    #Remove features with zero variance (i.e. constant)    
    def removeZeroVariance(data_frame):
        n_features_originally = data_frame.shape[1]
        selector = VarianceThreshold()
        selector.fit(data_frame)
        # Get the indices of zero variance feats
        feat_ix_keep = selector.get_support(indices=True)
        orig_feat_ix = np.arange(data_frame.columns.size)
        feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
        # Delete zero variance feats from the original pandas data frame
        data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],
                                     axis=1)
        # Print info
        n_features_deleted = feat_ix_delete.size
        print("  - Deleted %s / %s features (~= %.1f %%)" % (
            n_features_deleted, n_features_originally,
            100.0 * (np.float(n_features_deleted) / n_features_originally)))
        return data_frame
    #remove duplicate columns    
    def duplicate_columns(train,test):
        remove = []
        c = train.columns
        for i in range(len(c)-1):
            v = train[c[i]].values
            for j in range(i+1,len(c)):
                if np.array_equal(v,train[c[j]].values):
                    remove.append(c[j])
        train.drop(remove, axis=1, inplace=True)
        test.drop(remove,axis=1,inplace=True)
        return train,test
    #find correlated variables
    def findCorrelatedVariables(train,test,value,*saved):
        corrDic={}
        corr=train.corr()
        for i in range(len(corr)):
            for j in range(len(corr.iloc[i])):
                if j!=i:  #ignore same variable
                    if corr.iloc[i][j] > value:
                        #check currVar in dic. if not, add it to dic.  then, check if secondVar in dic. else add to dic.
                        if corr.index[i] not in corrDic.keys():
                            corrDic[corr.index[i]] = {corr.index[j]:corr.iloc[i][j]}
                        else: #if first var exists, check if second in dic. make new one
                            if corr.index[j] not in corrDic[corr.index[i]].keys():
                                corrDic[corr.index[i]][corr.index[j]] = corr.iloc[i][j]
                            else:
                                raise Exception('Error in preprocessing: finding correlated variables') #catch unexpected cases
        #store for future use
        if not saved:
            joblib.dump(corrDic, 'corrDic.pkl')
        return corrDic
    
    """
    Remove correlated variables. No improvement, dont use.
    """
    def removeCorrelatedVariables(train,test,corrDic,value,removeLimit):
        #find all variables in corrDic with at least minNumber of correlated features, and remove them
        variables = []
        currMax = 0
        #find variables correlated with maximum number of other features first, then remove them.
        for key,item in corrDic.items():
            if len(item) >= removeLimit and len(item) > currMax:
                currMax = len(item)
                
        for key,item in corrDic.items():
            if len(item) == currMax:
                variables.append(key)
        
        if len(variables) == 0:
            return train,test
            
        train.drop(variables,inplace=True,axis=1)
        test.drop(variables,inplace=True,axis=1)
        newCorrelations = findCorrelatedVariables(train,test,value,1)
        return removeCorrelatedVariables(train,test,newCorrelations,value,removeLimit)
        
    print('Beginning preprocessing.')
    print('Removing constant columns..')
    df,df2 = removeConstantsTrainTest(df,df2)
    print('Removing duplicate columns..')
    df,df2 = duplicate_columns(df,df2)
#    print('Finding correlated variables with value greater than '+str(correlationThreshold)+'..')
#    corrDic = findCorrelatedVariables(df,df2,correlationThreshold)
#    print('Removing correlated variables: \n 1)Value greater than '+str(correlationThreshold)+'\n 2)Correlated to at least '+str(cutTillNum)+'variables')
#    df,df2 = removeCorrelatedVariables(df,df2,corrDic,correlationThreshold,cutTillNum)
    return df,df2

#Split dataframe to data and target columns
def splitDatasetTarget(df3):
    dataset = df3.drop(['ID','TARGET'], axis=1).values
    target = df3['TARGET'].values
    return dataset,target

#split data and target columns to training and testing data.
def splitDataset(dataset,target,testsize=0.10):
    trainx, testx, trainy, testy = cross_validation.train_test_split(
        dataset, target, test_size=testsize)
        
    return trainx,testx,trainy,testy

#Run XGBOOST and create submission.csv
def runXGBoost():
    train = pd.read_csv('E:/Python projects and stuff/Dataset folder/Satander customer satisfaction/train.csv')
    test = pd.read_csv('E:/Python projects and stuff/Dataset folder/Satander customer satisfaction/test.csv')
    train,test=preProcess(train,test,0.7,5)
    #try normalizing. FAILED.
    #test= (test - test.mean()) / (test.max() - test.min())
    dataset,target = splitDatasetTarget(train)
    #dataset= (dataset - dataset.mean()) / (dataset.max() - dataset.min())

    trainx,testx,trainy,testy= splitDataset(dataset,target,0.1)
    clf = XGBClassifier(
                         learning_rate =0.02,
                         n_estimators=700,
                         max_depth=6,
                         min_child_weight=6,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.7,
                         reg_alpha=1.1,
                         objective= 'binary:logistic',
                         nthread=7,
                         scale_pos_weight=1,
                         seed=27)
    
   # X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)
    
    clf.fit(trainx,trainy, early_stopping_rounds=30, eval_metric = "auc",eval_set = [(testx,testy)])
    print('Overall AUC:', roc_auc_score(target, clf.predict_proba(dataset)[:,1]))
    testX = test.drop(['ID'], axis=1).values
    pred = clf.predict_proba(testX)[:,1]
    submission = pd.DataFrame({"ID":test['ID'], "TARGET":pred})
    submission.to_csv("xgboostdscrunGridSearch.csv", index=False)

if __name__ == '__main__':
    runXGBoost()
