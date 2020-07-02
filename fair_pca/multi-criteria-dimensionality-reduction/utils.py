import numpy as np
from standard_PCA import std_PCA

#use log to avoid the case product can overflow the floating point representation
def geo_mean_through_log(numberList):
    #if some is 0, return 0.
    if (np.amin(numberList) <= 1.e-12):
        return 0;
    
    logNumberList = np.log(numberList)
    return np.exp(logNumberList.sum()/len(numberList))

#data preprocessing helper methods
def scaleVar(dataframe,colArray):
    '''
    Normalize columns "together", meaning that I take the S.D. and mean to be the combined of all those columns.
    Ts makes more sense if columns are similar in meaning. For example, amount paid for 6 months as 6 months. 
    I normalize all 6 with the same variance.
    Example of usage:
        scaleVar(df.columns.values[2:4]) # scale the 3,4,5th columns together by dividing with the same variance
    '''
    SD = dataframe[colArray].stack().std(); #compute overall S.D.
    if SD == 0: #all number are the same. No need to do anything
        return;
    dataframe[colArray] = dataframe[colArray]/SD
    
def scaleVarOneCol(dataframe,nameStr):
    '''
    Given the name of one column, scale that column so that the S.D. is 1. The mean remains the same. For example,
    
    df = pandas.read.csv("some_path")
    scaleVar(df,feature1)
    '''
    if dataframe[nameStr].std() == 0: #all number are the same. No need to do anything
        return;
    dataframe[nameStr] = dataframe[nameStr]/dataframe[nameStr].std()
    
#input check
def input_check(n,k,d,B,function_name='the function'):
    '''
    Check that B is a list of k matrices of size n x n, and that d <= n.
    '''
    if (isinstance(function_name, str) == False):
        print("Error: check_input is used with function name that is not string. Exit the check.")
        return 1
        
    if (k<1):
        print("Error: " + function_name + " is called with k<1.")
        return 2
        
    if (len(B) < k):
        print("Error: " + function_name + " is called with not enough matrices in B.")
        return 3
        
    #check that matrices are the same size as n
    for i in range(k):
        if (B[i].shape != (n,n)):
            print("Error: " + function_name + " is called with input matrix B_i not the correct size." + "Note: i=" + str(i) + " , starting indexing from 0 to k-1")
            return 4
        
    if (((d>0) and (d<=n)) == False):
        print("Error: " + function_name + " is called with invalid value of d, which should be a number between 1 and n inclusive.")
        return 5
              
    return 0 #no error case

def getObj(n,k,d,B,X):
    """
    Given k PSD n-by-n matrices B1,...,Bk, and a projection matrix X which is n-by-n, give variance and loss of each group i.
    Additionally, compute max min variance, min max loss, Nash Social Welfare, and total variance objective by this solution X. 
    The matrix B_i should be centered (mean 0), since the formula to calculate variance will be by the dot product of B_i and X
    Arguments:
        n: original number of features (size of all B_i's)
        k: number of groups
        d: the target dimension
        B: list of PSD matrices, as numpy matrices. It must contain at least k matrices. If there are more than k matrices provided, the first k will be used as k groups.
        X: given solution (bad notation!! i cannot bear with it any more...). This method will still work even if not PSD or symmmetric or wrong rank as long as X has a correct dimension
        
    Return: a dictionary with keys 'Loss', 'Var', and 'Best' (for the best possible PCA in that group as if other group does not exist) to each group, 
    and three objectives MM_Var, MM_Loss, and NSW.
    """
    #input check
    if (input_check(n, k, d, B, function_name='fairDimReductionFractional') > 0):
        return -1
    #rank check
    if (np.linalg.matrix_rank(X) != d):
        print("Warning: getObj is called with X having rank not equal to d.")
        
    obj = dict() 
    
    # np.multiply is element-wise multiplication, np.sum(np.multiply) is computing <A, B>
    # \|A, P\|_F^2 = <A^T A, P P^T>
    best = [np.sum(np.sort(np.linalg.eigvalsh(B[i]))[-d:]) for i in range(k)]
    loss = [np.sum(np.multiply(B[i],X)) - best[i]  for i in range(k)]
    var = [np.sum(np.multiply(B[i],X)) for i in range(k)]
    
    #welfare objective
    obj.update({'MM_Var':np.amin(var),'MM_Loss':np.amin(loss),'NSW':geo_mean_through_log(var),'Total_Var':np.sum(var)})
    
    #Loss, Var, and Best to each group
    for i in range(k):
        obj.update({'Loss'+str(i):loss[i],'Var'+str(i):var[i],'Best'+str(i):best[i]})
        
    return obj  

def get_recon_error(n, k, d, data, projection):
    """
    Arguments:
        n: original number of features (size of all B_i's)
        k: number of groups
        d: the target dimension
        data: original data (x, n)
        projection matrix:  (n, d) for the target dimension d
    """
    err = []
    for i in range(k):
        err.append(np.linalg.norm(data[i]-np.dot(np.dot(data[i], projection), projection.T), 'fro')**2 / len(data[i]))
    return err

def get_optimal_error(n, k, d, data):
    
    optimal = []
    for i in range(k):
        P_all = std_PCA(data[i].T @ data[i] / len(data[i]), len(data[i][0])) 
        p = P_all[:,:d] @ P_all[:,:d].T
        optimal.append(np.linalg.norm(data[i]-np.dot(data[i], p), 'fro')**2 / len(data[i]))
    return optimal
        

def get_trace(n, k, d, data, projection):
    """
    Arguments:
        n: original number of features (size of all B_i's)
        k: number of groups
        d: the target dimension
        data: original data (x, n)
        projection matrix:  (n, d) for the target dimension d
    """
    tr = []

    for i in range(k):
        a = np.dot(projection.T, data[i].T)
        tr.append(np.trace(-np.dot(np.dot(a, data[i]), projection)/len(data[i]))) 
    return tr
    
