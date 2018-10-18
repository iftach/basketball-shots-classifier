import numpy as np
import scipy as sc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Calculate the general probabilities of making and missing a shot in the train set
def getShotProbs(train):
    missed = 0
    size = len(train)
    for _, row in train.iterrows():
        if row['FGM'] == 0:
            missed += 1
    return missed/size, (size-missed)/size

# The naive calssifier code
# recives train and test and returns the accuracy on the test 
def naiveClf(train, test):
    missed, made = getShotProbs(train)
    classifyBy = 0 if missed > made else 1
    correct = 0
    for _, row in test.iterrows():
        if row['FGM'] == classifyBy:
            correct += 1
    return correct/len(test)

# Gets a list of predictions from a classifier and a list of the actual results
# and returns the number of correct predictions
def getNumCorrect(pred, res):
    correct = 0
    if len(pred) != len(res):
        raise Exception("pred and res not same length")
    for i in range(len(pred)):
        if pred[i] == res[i]:
            correct += 1
    return correct

# The main function for this file that runs the naive classifier
# and the random forest classifier 10 times and returns the
# averaged accuracy over those 10 runs
def noHmmBase(df):
    features = ['GAME_ID', 'LOCATION', 'W','FINAL_MARGIN',
       'SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE',
       'CLOSEST_DEFENDER_PLAYER_ID', 'CLOSE_DEF_DIST', 'player_id']

    naive_total, rf_total = 0,0
    rf = RandomForestClassifier(n_estimators=1000)    
    for _ in range(10):
        train, test = train_test_split(df, test_size=0.2)
        rf.fit(train[features], train['FGM'])
        pred = rf.predict(test[features])
        rf_total += getNumCorrect(pred, list(test['FGM']))/len(pred)
        naive_total += naiveClf(train, test)
    print('Naive classifier: ', naive_total/10)
    print('Random Forest: ', rf_total/10)

if __name__=="__main__":
    df = pd.read_csv('data.csv')
    noHmmBase(df)
    