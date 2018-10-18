import numpy as np
import scipy as sc
import pandas as pd
from sklearn.model_selection import train_test_split

from hmm_base import createEmissionMatrix, createTRMatrix, calculateFirstShotProbs, getNumCorrect, createOseq, viterbi, runHmm
from no_hmm_base import getShotProbs

# Same as runHmm from hmm_base.py only that instead of
# general state probabilities each data set has its own
# initial probabilities
def runAdvanceHmm(groups, probs_dict, Tmatrix, Ematrix):
    total, correct = 0, 0
    for key, data in groups:
        Oseq, res = createOseq(data)
        vres = viterbi(probs_dict[key], Oseq, Tmatrix, Ematrix)
        correct += getNumCorrect(vres, res)
        total += len(res)
    return correct/total

# Parameters:
#   df - datafarme to use
#   col_name - a column name in the dataframe to group the data based on her
# Returns:
#   probability for each data set
def calcProbsDict(df, col_name):
    groups = df.groupby(col_name)
    probs = {}
    for key, data in groups:
        probs[key] = getShotProbs(data)
    return probs

# Parameters:
#   test - the test dataframe
#   by - list of columns to group by
# Returns:
#   list of lowest level groups
# Example:
#   if by is ['GAME_ID','player_id'] then we will get
#   a list with sequences of player shots from a certain game
def createSeqs(test, by):
    groups = test.groupby(by[0])
    by_ = by[1:]
    while by_ != []:
        tmp = []
        for _, group in groups:
            tmp = tmp + list(group.groupby(by_[0]))
        groups = tmp
        by_=by_[1:]
    return groups

# The main function for this file
# The function prints an averaged accuracy over 10 runs
# of various sequences
def advancePart(df):
    keys = ['game', 'game player', 'game player period', 'game period', 'game period pos',
            'game pos', "game player w player probs", 'game player period w period probs',
            'period', 'period player', 'player', 'player period', 'pts', 'pts player',
            'pts period', 'pts game', 'pts 2', 'pts 2 player', 'pts 2 period', 'pts 2 game', 'pts 3',
            'pts 3 player', 'pts 3 period', 'pts 3 game']

    total = {}

    for key in keys:
        total[key] = 0

    for _ in range(10):
        train, test = train_test_split(df, test_size=0.2)
        Ematrix = createEmissionMatrix(9,7,9,8,train)
        Tmatrix = createTRMatrix(train, 1)
        game_probs = calculateFirstShotProbs(train)
        player_probs = calcProbsDict(train,'player_id')
        period_probs = calcProbsDict(train,'PERIOD')
        pts_probs = calcProbsDict(train,'PTS_TYPE')

        total["game"] += runHmm(createSeqs(test,['GAME_ID']), game_probs, Tmatrix, Ematrix)
        total["game player"] += runHmm(createSeqs(test,['GAME_ID','player_id']), game_probs, Tmatrix, Ematrix)
        total["game player period"] += runHmm(createSeqs(test,['GAME_ID','player_id','PERIOD']), game_probs, Tmatrix, Ematrix)
        total["game period"] += runHmm(createSeqs(test,['GAME_ID','PERIOD']), game_probs, Tmatrix, Ematrix)
        total["game period pos"] += runHmm(createSeqs(test,['GAME_ID','PERIOD','POS']), game_probs, Tmatrix, Ematrix)
        total["game pos"] += runHmm(createSeqs(test,['GAME_ID','POS']), game_probs, Tmatrix, Ematrix)

        total["game player w player probs"] += runAdvanceHmm(createSeqs(test,['GAME_ID','player_id']), player_probs, Tmatrix, Ematrix)
        total['game player period w period probs'] += runAdvanceHmm(createSeqs(test,['GAME_ID','player_id','PERIOD']), period_probs, Tmatrix, Ematrix)

        total["period"] +=  runAdvanceHmm(createSeqs(test,['PERIOD']), period_probs, Tmatrix, Ematrix)
        total["period player"] +=  runAdvanceHmm(createSeqs(test,['PERIOD','player_id']), player_probs, Tmatrix, Ematrix)

        total["player"] += runAdvanceHmm(createSeqs(test,['player_id']), player_probs, Tmatrix, Ematrix)
        total["player period"] += runAdvanceHmm(createSeqs(test,['player_id','PERIOD']), period_probs, Tmatrix, Ematrix)

        total["pts"] += runAdvanceHmm(createSeqs(test,['PTS_TYPE']), pts_probs, Tmatrix, Ematrix)
        total["pts player"] += runAdvanceHmm(createSeqs(test,['PTS_TYPE','player_id']), player_probs, Tmatrix, Ematrix)
        total["pts period"] += runAdvanceHmm(createSeqs(test,['PTS_TYPE','PERIOD']), period_probs, Tmatrix, Ematrix)
        total["pts game"] += runHmm(createSeqs(test,['PTS_TYPE','GAME_ID']), game_probs, Tmatrix, Ematrix)
        total["pts 2"] += runHmm(createSeqs(test[test['PTS_TYPE']==2],['PTS_TYPE']), pts_probs[2], Tmatrix, Ematrix)
        total["pts 2 player"] += runAdvanceHmm(createSeqs(test[test['PTS_TYPE']==2],['PTS_TYPE','player_id']), player_probs, Tmatrix, Ematrix)
        total["pts 2 period"] += runAdvanceHmm(createSeqs(test[test['PTS_TYPE']==2],['PTS_TYPE','PERIOD']), period_probs, Tmatrix, Ematrix)
        total["pts 2 game"] += runHmm(createSeqs(test[test['PTS_TYPE']==2],['PTS_TYPE','GAME_ID']), game_probs, Tmatrix, Ematrix)

        total["pts 3"] += runHmm(createSeqs(test[test['PTS_TYPE']==3],['PTS_TYPE']), pts_probs[3], Tmatrix, Ematrix)
        total["pts 3 player"] += runAdvanceHmm(createSeqs(test[test['PTS_TYPE']==3],['PTS_TYPE','player_id']), player_probs, Tmatrix, Ematrix)
        total["pts 3 period"] += runAdvanceHmm(createSeqs(test[test['PTS_TYPE']==3],['PTS_TYPE','PERIOD']), period_probs, Tmatrix, Ematrix)
        total["pts 3 game"] += runHmm(createSeqs(test[test['PTS_TYPE']==3],['PTS_TYPE','GAME_ID']), game_probs, Tmatrix, Ematrix)

    print("Advance Part:")
    for key, value in total.items():
        print(key+": ", value/10)

if __name__=="__main__":
    df = pd.read_csv("data.csv")
    advancePart(df)
