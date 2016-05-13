# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:27:30 2016

@author: Thiru
"""
#==============================================================================
# Random Notes:
#
# FeatureEngineering REDUCES accuracy. Particularly likelyOutcomeForBreed. Strange.
#
#==============================================================================

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn import metrics
from sklearn.externals import joblib
import xgboost as xgb
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

# Load train and test
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

# For converting predicted numbers back to animal names
labelencoder = None

#==============================================================================
#                           Preprocessing methods
#==============================================================================

# Preprocess dataset.
# Input: 
# 1. <pandas dataframe> df (input data)
# 2. <boolean> train (if true, dataset is the training set. Else, testing dataset.)
def preprocess(df,train=True):
    ## standard preprocessing
    def removeRedundantColumns(df,train): #remove unwanted columns
        if train:
            colsToRemove = ['AnimalID','Name','OutcomeSubtype','DateTime']
        else:
            colsToRemove = ['DateTime','Name']
        return df.drop(colsToRemove,axis=1)
        
    def missingValues(df): #fill missing values#
        df['AgeuponOutcome'].fillna('0 years',inplace=True)
        df['SexuponOutcome'].fillna('Unknown',inplace=True)
        return df
        
    def oneHotEncode(df): #encode all categorical variables#
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        if train == True:
            columnsToEncode.remove('OutcomeType')
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df
        
    ##data specific preprocessing
    # create year/month/day
    def dateProcesser(df):
        dateTime = pd.DatetimeIndex(df['DateTime'])
        df['year'] = dateTime.year
        df['month'] = dateTime.month
        df['day'] = dateTime.day
        df['hour'] = dateTime.hour
        return df
    
    #convert name to boolean, yes 
    def nameProcesser(df): #this makes it alot worse :S
        name = df['Name']
        name[name.isnull() == False] = 1
        name[name.isnull()] = 0
        return df
    
    #in months
    def processAge(df):
        for i in df.index:
            currAge = df['AgeuponOutcome'][i]
            if 'year' in currAge or 'years' in currAge:
                position = currAge.find('year')
                newAge = int(currAge[:position-1])*12
            elif 'month' in currAge or 'months' in currAge:
                position = currAge.find('month')
                newAge = int(currAge[:position-1])
            elif 'weeks' in currAge or 'week' in currAge:
                position = currAge.find('week')
                newAge = (int(currAge[:position-1]))/4
            elif 'days' in currAge or 'day' in currAge:
                position = currAge.find('day')
                newAge = int(currAge[:position-1])/30
            else:
                raise Exception('An error occured with processAge')
            df['AgeuponOutcome'].set_value(i,newAge)
        df['AgeuponOutcome'] = df['AgeuponOutcome'].astype(float)
        return df
    
    def processColor(df):
        color2 = pd.Series(name= 'color2', index=df.index,dtype='object')
        for i in df.index:
            currColor = df['Color'][i]
            if '/' not in currColor:
                color2.set_value(i,'None')
            else:
                secondColor = currColor[currColor.find('/')+1:]
                firstColor = currColor[:currColor.find('/')]
                color2.set_value(i,secondColor)
                df['Color'].set_value(i,firstColor)
        df['color2'] = color2
        return df
        
    def customEncode(df):
        global labelencoder
        le = LabelEncoder()
        le.fit(df['OutcomeType'])
        df['OutcomeType'] = le.transform(df['OutcomeType'])
        labelencoder = le
        return df
        
    df = dateProcesser(df)
    #df = nameProcesser(df) #makes it alot worse for some reason
    df = removeRedundantColumns(df,train)
    df = missingValues(df)
    df = processAge(df)
    df = processColor(df)
    df = featureEngineering(df,train) #makes it worse too wtf.
    df = oneHotEncode(df)
    if train == True:
        df = customEncode(df)
    return df
#==============================================================================
#                           Feature Engineering
#==============================================================================
def featureEngineering(df,train):
    
    #changes age lesser or equal 12 months to be puppy, greater than to be Adult
    def babyOrAdult(df):
        col = pd.Series(name= 'adult', index=df.index)
        col[df['AgeuponOutcome'] > 12] = 'Adult'
        col[df['AgeuponOutcome'] <= 12] = 'Puppy'
        df['Adult'] = col
        return df
    # makes a 'popular breed' yes no boolean column
    def makePopularBreed(df):
        pop = Counter(df['Breed'])
        popCol = pd.Series(name='breedPop',index=df.index)
        for i in df.index:
            currPup = df['Breed'][i]
            popCol.set_value(i,pop[currPup])
        df['popCol'] = popCol
        return df
    
    #split to 2 columns. 
    # 1.canReproduce
    # 2.gender
    def makeGender(df):
        #gender - M / F / Unknown --> 1 / 0 / -99
        #reproduce Y / N / Unknown --> 1 / 0 / -99
        col = df['SexuponOutcome']
        gender = pd.Series(name= 'gender', index=df.index)
        canReproduce = pd.Series(name= 'reproduce', index=df.index)
        for i in df.index:
            sex = col[i]
            if sex == 'Unknown':
                gender.set_value(i,-99)
                canReproduce.set_value(i,-99)
            elif 'Female' in sex:
                gender.set_value(i,0)
                if 'Intact' in sex:
                    canReproduce.set_value(i,1)
                else:
                    canReproduce.set_value(i,0)
            elif 'Male' in sex:
                gender.set_value(i,1)
                if 'Intact' in sex:
                    canReproduce.set_value(i,1)
                else:
                    canReproduce.set_value(i,0)
            else:
                print('Error occured in makeGender at index '+str(i))
        df['gender'] = gender
        df['canReproduce'] = canReproduce
        return df
    
    def likelyOutcomeForBreed(df):
        nonlocal train
        if os.path.exists('pickle/likelyOutcome.pkl'):
            dic = joblib.load('pickle/likelyOutcome.pkl')
        else:
            if train == False:
                raise Exception('Run this with train csv first. likelyOutcome not generated')
            dic = {}
            Breeds = df['Breed'].unique()
            for breed in Breeds:
                counts = Counter(df[df['Breed'] == breed]['OutcomeType'])
                mostFreqOutcome = counts.most_common(1)[0][0]
                dic[breed] = mostFreqOutcome
            joblib.dump(dic,'pickle/likelyOutcome.pkl')
        
        df['likelyOutcome'] = df['Breed'].map(dic)
        df['likelyOutcome'].fillna('-99',inplace=True)
        return df
    df = babyOrAdult(df)
    df = makePopularBreed(df)
    df = makeGender(df)
    df = likelyOutcomeForBreed(df)
    return df

#==============================================================================
#                       Splitting to target and data
#==============================================================================
def splitDatasetTarget(df):
    dataset = df.drop(['OutcomeType'], axis=1)
    target = df['OutcomeType']
    return dataset,target

def splitDataset(dataset,target,testsize=0.2):
    trainx, testx, trainy, testy = cross_validation.train_test_split(
        dataset, target, test_size=testsize)
        
    return trainx,testx,trainy,testy
    
#==============================================================================
#                           Models
#==============================================================================
def xgBoost():
    clf = xgb.XGBClassifier(max_depth = 8,n_estimators=100,nthread=8,seed=1,silent=1,
                            objective= 'multi:softprob',learning_rate=0.1,subsample=0.9)
    return clf

def randomForest():
    clf = RandomForestClassifier(max_depth=6,n_jobs=8,n_estimators=500)
    return clf

#==============================================================================
#                           Tuning
#==============================================================================

#attain optimal num boosting rounds
def xgboostCV(clf, dataset,target ,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        print('Running XGBOOST cross validation')
        xgb_param = clf.get_xgb_params()
        xgb_param['num_class'] = 6
        xgtrain = xgb.DMatrix(dataset.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['mlogloss'], early_stopping_rounds=early_stopping_rounds, show_progress=False)
        CV_ROUNDS = cvresult.shape[0]
        print('Optimal Rounds: '+str(CV_ROUNDS))
        clf.set_params(n_estimators=CV_ROUNDS)
    
    print('Fitting to dataset')
    #Fit the algorithm on the data
    clf.fit(dataset, target,eval_metric='mlogloss')

    #Predict training set:
    dtrain_predictions = clf.predict(dataset)
    accuracy = metrics.accuracy_score(target,dtrain_predictions)
    print('CV Accuracy on training set: '+str(accuracy))
    return clf,CV_ROUNDS

#==============================================================================
#                           Predict Test
#==============================================================================
def predictTest(test,clf,csvname = 'submission.csv'):
    ID = test['ID']
    test = preprocess(test,train=False)
    test.drop('ID',axis=1,inplace=True)
    predictedTest = clf.predict_proba(test)
    columnNames = list(labelencoder.inverse_transform([0,1,2,3,4]))
    df = pd.DataFrame(predictedTest)
    for i in range(len(df.columns)):
        df = df.rename(columns={i: columnNames[i]})
    df['ID'] = ID
    cols = list(df.columns)
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(csvname,index=False)
    print('Test prediction completed')
        
        
if __name__ in '__main__':
    train = preprocess(train)
    dataset,target = splitDatasetTarget(train)
    trainx,testx,trainy,testy = splitDataset(dataset,target)
    clf = xgBoost()
    clf,CV_ROUNDS = xgboostCV(clf,dataset,target)
    predictTest(test,clf,csvname='submission.csv')


