import numpy as np
import scipy as sc
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing

from no_hmm_base import noHmmBase
from hmm_base import hmmBase
from advance_part import advancePart
from creative_part import creativePart

if __name__=="__main__":
    df = pd.read_csv("data.csv")

    # noHmmBase(df)   
    # hmmBase(df)
    # advancePart(df)
    # creativePart(df)

    multiprocessing.Process(target=noHmmBase, args=(df,)).start()
    multiprocessing.Process(target=hmmBase, args=(df,)).start()
    multiprocessing.Process(target=advancePart, args=(df,)).start()
    multiprocessing.Process(target=creativePart, args=(df,)).start()
