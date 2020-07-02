import pandas as pd
from utils import scaleVar, scaleVarOneCol

def preprocess_credit(path):
    '''
    Given a path to the credit data, will return a pandas datafrme with sacled variance.
    For most of columns, we scale the variance to 1. 
    Each of payCol, billCol, and payAMTCol have 6 features as a record over 6 billing period. We decided to scale them with the same normalization ratio.
    Hence, they may have variance different from 1 (but the variance over 6 columns will be scaled to 1)
    '''
    df = pd.read_csv(path);
    df = df.drop(columns=['ID']); #delete the ID row
        
    ##do the normalization, by section
    #singleCol = df.columns.values[0:list(df.columns.values).index('PAY_1')]
    ##for this I scale each col individually
    #for col in singleCol:
    #    scaleVarOneCol(df,col);
    #    
    ##the group part1
    #payCol = df.columns.values[list(df.columns.values).index('PAY_1'):list(df.columns.values).index('PAY_6')+1]
    #scaleVar(df,payCol)
    #
    ##the group part2
    #billCol = df.columns.values[list(df.columns.values).index('BILL_AMT1'):list(df.columns.values).index('BILL_AMT6')+1]
    #scaleVar(df,billCol)
    #
    ##the group part3
    #payAMTCol = df.columns.values[list(df.columns.values).index('PAY_AMT1'):list(df.columns.values).index('PAY_AMT6')+1]
    #scaleVar(df,payAMTCol)
    
    return df

def preprocess_income(folder):
    '''
    The data has 5 groups as A1.csv ... A5.csv. The path is the folder that contains all 5 files.
    Return: 
        [df1 df2 df3 df4 df5]
    where df[i] is the dataframe of group i
    Note that the input data has been normalized to have mean 0 and variance 1 in each dimension in each group
    '''
    df = []
    for i in range(5):
        df.append(pd.read_csv(folder+"/A"+str(i+1)+".csv", header=None))
        
    return df
    