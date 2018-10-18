import numpy as np
import scipy as sc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from hmm_base import createEmissionMatrix, createTRMatrix, calculateFirstShotProbs, createOseq, viterbi
from advance_part import calcProbsDict

# Parameters:
#   vres - predictions
#   res - real results
# Returns:
#   the tp and fn values from the confusion matrix
def getTpAndFn(vres, res):
    tp, fn = 0, 0
    for i in range(len(res)):
        if vres[i] == 1 and res[i] == 1:
            tp += 1
        elif vres[i] == 0 and res[i] == 1:
            fn += 1
    return tp, fn

# same as runHmm in hmm_base.py with the differece of returning averaged recall instead of accuracy
def runHmmWithRecall(data_sets, init_probs, Tmatrix, Ematrix):
    tp, fn = 0, 0
    for _,data in data_sets:
        Oseq, res = createOseq(data)
        vres = viterbi(init_probs, Oseq, Tmatrix, Ematrix)
        tmp_tp, tmp_fn = getTpAndFn(vres, res)
        tp += tmp_tp
        fn += tmp_fn
    return tp/(tp + fn)

# Parameters:
#   test - test set
#   init_probs - initial state probabilities per player
#   Tmatrix - transition matrix
#   Ematrix - emission matrix
# Returns:
#   recall value
def runCreativeClf(test, init_probs, Tmatrix, Ematrix):
    tp, fn = 0,0
    for _, row in test.iterrows():
        res = int(row['FGM'])
        prob0 = init_probs[row['player_id']][0]*Ematrix[0,row['SHOT_CLOCK_CAT'], row['CLOSE_DEF_DIST_CAT'], row['TOUCH_TIME_CAT'], row['SHOT_DIST_CAT']]
        prob1 = init_probs[row['player_id']][1]*Ematrix[1,row['SHOT_CLOCK_CAT'], row['CLOSE_DEF_DIST_CAT'], row['TOUCH_TIME_CAT'], row['SHOT_DIST_CAT']]
        pred = 0 if prob0 > prob1 else 1
        if pred == 1 and res == 1:
            tp += 1
        elif pred == 0 and res == 1:
            fn += 1
    return tp/(tp + fn)

# gets predictions and results and return the recall
def calcRecall(pred, res):
    if len(pred) != len(res):
        raise Exception('unequal pred and res len')
    tp, fn = 0, 0
    for i in range(len(pred)):
        if pred[i] == 1 and res[i] == 1:
            tp += 1
        elif pred[i] == 0 and res[i] == 1:
            fn += 1
    return tp/(tp + fn)

# Main function for this file
# runs the random forest classifier, the basic structured classifier and the creative classifier
# and prints the averaged recall for each of them over 10 runs
def creativePart(df):
    total_base, total_creative, total_rf = 0, 0, 0
    rf = RandomForestClassifier(n_estimators=1000)
    features = ['GAME_ID', 'LOCATION', 'W','FINAL_MARGIN',
       'SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE',
       'CLOSEST_DEFENDER_PLAYER_ID', 'CLOSE_DEF_DIST', 'player_id']
    for _ in range(10):
        train, test = train_test_split(df, test_size=0.2)
        player_probs = calcProbsDict(train, 'player_id')
        game_probs = calculateFirstShotProbs(train)
        Ematrix = createEmissionMatrix(9,7,9,8,train)
        Tmatrix = createTRMatrix(train, 1)

        rf.fit(train[features], train['FGM'])
        pred = rf.predict(test[features])
        total_rf += calcRecall(pred, list(test['FGM']))
        total_base += runHmmWithRecall(test.groupby('GAME_ID'),game_probs, Tmatrix, Ematrix)
        total_creative += runCreativeClf(test, player_probs, Tmatrix, Ematrix)
    
    print('Creative part: random forest recall: ', total_rf/10)
    print('Creative part: base hmm recall: ', total_base/10)
    print('Creative part: creative classifier recall: ', total_creative/10)


if __name__=="__main__":
    df = pd.read_csv("data.csv")
    creativePart(df)