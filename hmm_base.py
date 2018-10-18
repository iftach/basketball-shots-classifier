import numpy as np
import scipy as sc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from no_hmm_base import getNumCorrect


# The parameters are the number of categories for each feature
# and the train data
# the fuction returns the emission matrix used by the viterbi algorithm
def createEmissionMatrix(cat1, cat2, cat3, cat4,df):
    matrix = {}
    missed = 0
    made = 0
    for a in range (1,cat1+1):
        for b in range (1, cat2+1):
            for c in range (1, cat3+1):
                for d in range (1, cat4+1):
                    for e in range (0,2):
                        matrix[e,a,b,c,d] = 0
    for i, row in df.iterrows():
        matrix[int(row['FGM']),row['SHOT_CLOCK_CAT'],row['CLOSE_DEF_DIST_CAT'],row['TOUCH_TIME_CAT'],row['SHOT_DIST_CAT']]+=1
        if row['FGM'] == 0:
            missed += 1
        else:
            made +=1
    for key, value in matrix.items():
        if(key[0] == 0):
            matrix[key] = value/missed
        else:
            matrix[key] = value/made
    return matrix 

# vec is a list of 0 and 1
# the function treats the list as a binary number and
# returns the corresponding decimal number
def calcPos(vec):
    pos = 0
    for i in range(len(vec)):
        pos += vec[i]*(2**i)
    return pos

# Parametes:
#   df - dataframe object
#   numPrev - indicates which n-gram model is used, 1 for bigram, 2 for trigram and so on
# Returns:
#   the transition matrix used by the viterbi algorithm
def createTRMatrix(df, numPrev):
    matrix = np.zeros((2**numPrev,2), dtype=float)
    prev = []
    for i in range(numPrev):
        prev.append(int(df.iloc[i]['FGM']))
    sums = np.zeros(2**numPrev)
    pos = calcPos(prev)
    for i, row in df[numPrev:].iterrows():
        sums[pos] += 1
        curr = int(row['FGM'])
        matrix[pos,curr] += 1
        del prev[0]
        prev.append(curr)
        pos = calcPos(prev)
    for j in range(len(sums)):
        if sums[j] != 0:
            matrix[j,0] = matrix[j,0]/sums[j]
            matrix[j,1] = matrix[j,1]/sums[j]
    return matrix

# Parameters: a datafram object
# Returns: the probabilities that a first shot in a game will
#           be a made or a missed
def calculateFirstShotProbs(df):
    grouped_df = df.groupby('GAME_ID')
    count_total = 0 
    count_made = 0
    count_missed = 0
    for key, data in grouped_df:
        sorted_data = data.sort_values(by=['PERIOD', 'GAME_CLOCK'], ascending=[True, False])
        if int(sorted_data.iloc[0]['FGM']) == 1:
            count_made += 1
        else:
            count_missed += 1
        count_total += 1
    return (count_missed/count_total, count_made/count_total)

# Parameters:
#   stateSpace - the state space used in our model
#   T1Column - a column from the T1 matrix from the viterbi algorithm
#   TMatrixColumn - a column from the transition matrix
#   emissionValue - value from a cell in the emission matrix
# Returns:
#   the highest probability and the corresponding state
def getMaxValueAndStateForViterbiBase(stateSpace, T1Column, TMatrixColumn, emissionValue):
    maxValue = 0
    maxState = stateSpace[0]
    for k in range(len(stateSpace)):
        currValue = T1Column[k]*TMatrixColumn[k]*emissionValue
        if(currValue > maxValue):
            maxValue = currValue
            maxState = stateSpace[k]
    return (maxValue, maxState)

# Parameters:
#   initialProbs - the initial state probabilities
#   Osequence -  a sequence of observations
#   Tmatrix - the transition matrix
#   Ematrix - the emission matrix
#   stateSpace - the state space of the model
#   observeSpaceSize - the obsevation space size
# Returns:
#   list of most likely states path
def viterbi(initialProbs, Osequence, Tmatrix, Ematrix,stateSpace = [0,1], observeSpaceSize=9*7*9*8):
    stateSize = len(stateSpace)
    T1 = np.zeros((stateSize, len(Osequence)), dtype=float)
    T2 = np.empty((stateSize, len(Osequence)), dtype=int)
    state = np.zeros(2, dtype=float)
    for i in range(stateSize):
        T1[i,0] = initialProbs[i]*Ematrix[i,Osequence[0][0],Osequence[0][1],Osequence[0][2],Osequence[0][3]]
        T2[i,0] = -1
    for i in range(1,len(Osequence)):
        for j in range(stateSize):
            value, state[j] = getMaxValueAndStateForViterbiBase(stateSpace, T1[:,i-1],
                        Tmatrix[:,j],Ematrix[j,Osequence[i][0],Osequence[i][1],Osequence[i][2],Osequence[i][3]])
            T1[j,i] = value
            T2[j,i] = state[j]
    maxStateArray = np.empty(len(Osequence), dtype=int)
    maxStateArray[-1] = T1[:,-1].argmax()
    for i in reversed(range(1,len(Osequence))):
        maxStateArray[i-1] = T2[maxStateArray[i],i]
    return maxStateArray

# Parameters:
#   df - dataframe with the records of the sequence
# Returns:
#   Oseq - a list of vectors, one for each record features
#   res - list of results, one for each record result
def createOseq(df):
    Oseq = []
    res = []
    sorted_ = df.sort_values(by=['GAME_ID','PERIOD','GAME_CLOCK'], ascending=[True,True,False])
    for i, row in sorted_.iterrows():
        Oseq.append((row['SHOT_CLOCK_CAT'], row['CLOSE_DEF_DIST_CAT'], row['TOUCH_TIME_CAT'], row['SHOT_DIST_CAT']))
        res.append(int(row['FGM']))
    return Oseq, res

# Parameters:
#   data_sets - dataframe sets NEEDS TO BE THE RESULT OF A DATAFRAM GROUPBY
#   initProbs - the initial state probabilities
#   Tmatrix - transition matrix for the viterbi algorithm
#   Ematrix - emission matrix for viterbi
# Returns:
#   the averaged accuracy over all the given dataframes
def runHmm(data_sets, initProbs, Tmatrix, Ematrix):
    total, correct = 0, 0
    for _,data in data_sets:
        Oseq, res = createOseq(data)
        vres = viterbi(initProbs, Oseq, Tmatrix, Ematrix)
        correct += getNumCorrect(vres, res)
        total += len(res)
    return correct/total

# The main function for this file
# runs the hmm classifier 10 times and returns the averaged accuracy
def hmmBase(df):
    total = 0
    for _ in range(10):
        train, test = train_test_split(df, test_size=0.2)
        init_probs = calculateFirstShotProbs(train)
        Ematrix = createEmissionMatrix(9,7,9,8,train)
        Tmatrix = createTRMatrix(train, 1)
        total += runHmm(test.groupby('GAME_ID'),init_probs, Tmatrix, Ematrix)
    print('Base Part Hmm: ', total/10)

if __name__=="__main__":
    df = pd.read_csv("data.csv")
    hmmBase(df)