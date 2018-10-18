import numpy as np
import scipy as sc
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from math import ceil, floor
from sklearn.model_selection import train_test_split
import string
import threading
import multiprocessing
import csv
import warnings


def getCatShotClock(time):
    if(0 <= time and time <= 2):
        return 1
    elif(2 < time and time <= 4):
        return 2
    elif(4 < time and time <=8):
        return 3
    elif(8 < time and time <= 13):
        return 4
    elif(13 < time and time <= 18):
        return 5
    elif(18 < time and time <= 20):
        return 6
    elif(20 < time and time <= 22 ):
        return 7
    elif(20 < time and time <= 24):
        return 8
    else:
        return 9

def getCatTouchTime(time):
    if(0 <= time and time <= 0.1):
        return 1
    elif(0.1 < time and time <= 0.3):
        return 2
    elif(0.3 < time and time <= 0.7):
        return 3
    elif(0.7 < time and time <= 0.9):
        return 4
    elif(0.9 < time and time <= 1.5):
        return 5
    elif(1.5 < time and time <= 3):
        return 6
    elif(3 < time and time <= 7):
        return 7
    elif(7 < time and time <= 24):
        return 8
    else:
        return 9

def getCatClosestDef(dist):
    if(0 <= dist and dist <= 1):
        return 1
    elif(1 < dist and dist <= 3):
        return 2
    elif(3 < dist and dist <=   5):
        return 3
    elif(5 < dist and dist <= 8):
        return 4
    elif(8 < dist and dist <= 10):
        return 5
    elif(10 < dist and dist <= 80):
        return 6
    else:
        return 7

def getCatShotDist(dist):
    if(0 <= dist and dist <= 5):
        return 1
    elif(5 < dist and dist <= 9):
        return 2
    elif(9 < dist and dist <= 13):
        return 3
    elif(13 < dist and dist <= 17):
        return 4
    elif(17 < dist and dist <= 21.8):
        return 5
    elif(21.8 < dist and dist <= 25):
        return 6
    elif(25 < dist):
        return 7
    else:
        return 8

def createCatgorizeCSV():   
    df= pd.read_csv('shot_logs_no_nan.csv')
    df["SHOT_CLOCK_CAT"]=1
    df["CLOSE_DEF_DIST_CAT"]=1
    df["TOUCH_TIME_CAT"]=1
    df["SHOT_DIST_CAT"]=1
    df["SHOT_RESULT_CAT"]=1
    print(getCatShotClock(0.2))

    count = 0
    for i, row in df.iterrows():
        if(row['SHOT_RESULT'] == "missed"):       
            df['SHOT_RESULT_CAT'][i] = 0
        else:
            df['SHOT_RESULT_CAT'][i] = 1
        df['SHOT_CLOCK_CAT'][i] = getCatShotClock(row['SHOT_CLOCK'])
        df['CLOSE_DEF_DIST_CAT'][i] = getCatClosestDef(row['CLOSE_DEF_DIST'])
        df['SHOT_DIST_CAT'][i] = getCatShotDist(row['SHOT_DIST'])
        df["TOUCH_TIME_CAT"][i] = getCatTouchTime(row['TOUCH_TIME']) 
    
        count += 1
        if(count%100==0):
            print(count)

    df.to_csv('shot_log_cat_no_nan.csv')

def editGameClock():
    fulldf=[]
    with open('shot_log_cat_no_nan.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                row['GAME_CLOCK'] = int(row['GAME_CLOCK'].split(':')[0])*60+int(row['GAME_CLOCK'].split(':')[1])
                fulldf.append(row)
    df_full = pd.DataFrame(fulldf)
    df_full.to_csv('shot_log_cat_no_nan2.csv')

def addCloseDefDistShotDistRatio():
    count = 0
    df = pd.read_csv('shot_log_cat_no_nan2.csv')
    df['DEF_DIST_RATIO'] = 0.0
    for i, row in df.iterrows():
        if(row['CLOSE_DEF_DIST'] != 0):
            df['DEF_DIST_RATIO'][i] = row['SHOT_DIST']/row['CLOSE_DEF_DIST']
        else:
            df['DEF_DIST_RATIO'][i] = row['SHOT_DIST']/(row['CLOSE_DEF_DIST']+0.1) 
        count += 1
        if(count%100==0):
            print(count)
    df.to_csv('shot_log_cat_no_nan_with_ratio.csv')

def addCatDefDistRatio():
    df = pd.read_csv('shot_log_cat_no_nan_with_ratio.csv')
    df['DEF_DIST_RATIO_CAT'] = 1
    df.to_csv('tmp.csv')
    fulldf=[]
    with open('tmp.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                row['DEF_DIST_RATIO_CAT'] = int(getCatDefDistRatio(float(row['DEF_DIST_RATIO'])))
                fulldf.append(row)
    df_full = pd.DataFrame(fulldf)
    df_full.to_csv('shot_log_cat_no_nan_with_ratio_cat.csv')

def getCatDefDistRatio(ratio):
    if(0 <= ratio and ratio < 1):
        return 1
    elif(1 <= ratio and ratio <= 2):
        return 2
    elif(2 < ratio and ratio <= 3.5):
        return 3
    elif(3.5 < ratio and ratio <= 5):
        return 4
    elif(5 < ratio and ratio <= 6):
        return 5
    elif(6 < ratio and ratio <= 8):
        return 6
    elif(8 < ratio):
        return 7
    else:
        return 8


def getCatDrib(drib):
    if(0 == drib):
        return 1
    elif(1 == drib):
        return 2
    elif(2 == drib):
        return 3
    elif(4 == drib):
        return 4 
    else:
        return 5 




def addCatDribells():
    fulldf=[]
    df = pd.read_csv('shot_log_cat_no_nan_with_ratio_cat.csv')
    df['DRIB_CAT'] = 1
    df.to_csv('tmp.csv')
    with open('tmp.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                row['DRIB_CAT'] = int(getCatDrib(int(row['DRIBBLES'])))
                fulldf.append(row)
    df_full = pd.DataFrame(fulldf)
    df_full.to_csv('shot_log_cat_ratio_drib.csv')

def getPlayerPos():
    df = pd.read_csv('dfifull_no_nan_pos.csv')
    pos = {}
    players = df.groupby('player_id')
    for key, data in players:
        pos[key] = data.iloc[0]['position']
    return pos

def getCatPos(pos):
    if(pos == 'G' or pos =='SG'):
        return 1
    elif( pos == 'SF' or pos == 'PG'):
        return 2
    elif(pos == 'PF' or pos == 'F'):
        return 3 
    elif(pos == 'C'):
        return 4
    else:
        return 5

def addPos():
    fulldf=[]
    df = pd.read_csv('shot_log_cat_ratio_drib.csv')
    df['POS'] = '5'
    df.to_csv('tmp.csv')
    pos_dict=getPlayerPos()
    with open('tmp.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if int(row['player_id']) in pos_dict.keys():
                    row['POS'] = int(getCatPos(pos_dict[int(row['player_id'])]))
                fulldf.append(row)
    df_full = pd.DataFrame(fulldf)
    df_full.to_csv('shot_log_cat_ratio_drib_pos.csv')

warnings.filterwarnings('ignore')
#createCatgorizeCSV() 
#editGameClock()
#addCloseDefDistShotDistRatio()
#addCatDefDistRatio()
#addCatDribells()
addPos()


#df = pd.read_csv('dfifull_no_nan_pos.csv')
#print(set(df['position']))