import pandas as pd
from utils import input_check

#for optimization
from cvxopt import matrix

#optimization part
import numpy as np
import cvxopt as cvx
import picos as pic

from numpy import array

#use log to avoid the case product can overflow the floating point representation
def geo_mean_through_log(numberList):
    #if some is 0, return 0.
    if (np.amin(numberList) <= 1.e-12):
        return 0;
    
    logNumberList = np.log(numberList)
    return np.exp(logNumberList.sum()/len(numberList))

#the SDP for fair PCA method

def fairDimReductionFractional(n,k,B,list_n='one',Obj='MM_Loss',list_d='all',verbose=1,print_other_obj=True,return_option='run_statistics',save=True,savedPath='fairDimReductionFractional.csv'):
    """
    Given k PSD n-by-n matrices B1,...,Bk, solve the (fractional) convex optimization of fair dimensional reduction.
    Arguments:
        k: number of groups
        n: original number of features (size of all B_i's)
        B: list of PSD matrices, as numpy matrices. It must contain at least k matrices. If there are more than k matrices provided, the first k will be used as k groups.
        list_n: by default, this is simply n, which is the total number of features. 
                If 'all', this n0 (number of dimension, first n0 features are used) ranges from d0+1 to n (d0 is the target dimension of that iteration)
                Else, you can specify as a list of n_0.
        list_d: list of target dimensions to project to. By default ('all'), it is from 1 to n-1.
        print_other_obj:setting to True will also print other welfare economic objective (total of four, including one specified as input Obj)
        verbose: set to 1 if the details are to be printed. Set to 2 to print the information table of each iteration
        save: will save to csv if set to True
        savedPath: path of the file to export the result to.
        Obj:the objective to optimize. Must be MM_Var (maximize the minimum variance), MM_Lose (default) (minimize the maximum loss, output the negative), or NSW (Nash social welfare)
        return_option:  by default, it returns the runtime, n,d, the rank, and several objectives of each group. 
                        Another option 'frac_sol' is to return a list of fractional solution X. list_X (the additional output) will be a list, each row containing one run of particular value of n and d.
                        In each row, it contains value of n,d and solution X as cvx matrix. One can convert this back to numpy by:
                            import numpy as np
                            array(list_X[0][2]) --> this gives numpy solution matrix X of the first setting of (n,d).
                            array(list_X[i][2]) --> this gives numpy solution matrix X of the (i+1)th setting of (n,d).
        
    """
    
    #input check
    if (input_check(n, k, 1, B, function_name='fairDimReductionFractional') > 0):
        return -1
        
    #for storing results of the optimization
    runstats = pd.DataFrame()
    if (return_option == 'frac_sol'):
        list_X = []
    
    #list of all d
    if (list_d == 'all'):
        list_d = range(1,n)

    for d in list_d:
        #valid value of n_0
        if (list_n == 'one'):
            list_n_this_d = [n]
        elif (list_n == 'all'):
            list_n_this_d = range(d+1,n+1)
        else:
            list_n_this_d = list_n
            
        for n0 in list_n_this_d:
            #shorten version of the matrix, in case we want to delete any earlier features for experiments
            Bnumpy_s = [B[i][np.ix_(range(0,n0),range(0,n0))] for i in range(k)]

            #now define the problem
            B_s = [matrix(B[i][np.ix_(range(0,n0),range(0,n0))]) for i in range(k)]

            fairPCA = pic.Problem()
            n = n0

            I =pic.new_param('I',cvx.spmatrix([1]*n,range(n),range(n),(n,n))) #identity matrix

            # Add the symmetric matrix variable.
            X=fairPCA.add_variable('X',(n,n),'symmetric') #projection matrix, should be rank d but relaxed
            z=fairPCA.add_variable('z',1) #scalar, for the objective

            # Add parameters for each group
            A = [pic.new_param('A'+str(i),B_s[i]) for i in range(k)]

            #best possible variance for each group
            best = [np.sum(np.sort(np.linalg.eigvalsh(Bnumpy_s[i]))[-d:]) for i in range(k)]
            
            # Constrain X on trace
            fairPCA.add_constraint(I|X<=d)

            # Constrain X to be positive semidefinite.

            fairPCA.add_constraint(X>>0)
            fairPCA.add_constraint(X<<I)
            
            #the following depends on the type of the problems. Here we coded 3 of them: 
            #1) max min variance 2) min max loss 3) Nash social welfare of variance
            
            if(Obj=='MM_Loss'):
                # Add loss constriant
                fairPCA.add_list_of_constraints([(A[i]|X) - best[i] >= z for i in range(k)]) #constraints

                # Set the objective.
                fairPCA.set_objective('max',z)
                
            elif (Obj=='MM_Var'):
                # Add variance constriant
                fairPCA.add_list_of_constraints([(A[i]|X) >= z for i in range(k)]) #constraints

                # Set the objective.
                fairPCA.set_objective('max',z)
                
            elif (Obj=='NSW'):
                s=fairPCA.add_variable('s',k) #vector of variances
                # Add variance constriant
                fairPCA.add_list_of_constraints([(A[i]|X) >= s[i] for i in range(k)]) #constraints

                # Set the objective.
                fairPCA.add_constraint( z <= pic.geomean(s) )
                fairPCA.set_objective('max',z)
                
            else:
                fairPCA.set_objective('max',z)
                print("Error: fairDimReductionFractional is called with invalid Objective. Supported Obj augements are: ... Exit the method")
                return

            solveInfo=fairPCA.solve(verbose = 0,solver='cvxopt')

            var = [np.sum(np.multiply(Bnumpy_s[i],X.value)) for i in range(k)]
            loss = [var[i] - best[i] for i in range(k)]
            
            #print(solveInfo)
            #dictionary of info for this iterate
            solveInfoShort = {}
            #for key in ('time','obj','status'):
            #    solveInfoShort[key] = solveInfo[key]
                
            if (print_other_obj):
                solveInfoShort.update({'MM_Var':np.amin(var),'MM_loss':np.amin(loss),'NSW':geo_mean_through_log(var),'Total_Var':np.sum(var)})
            
            #solveInfoShort.update({'n':n0,'gap':solveInfo['cvxopt_sol']['gap'],'d':d,'rank':np.linalg.matrix_rank(array(X.value),tol=1e-6,hermitian =True)})
            
            for i in range(k):
                solveInfoShort.update({'Loss'+str(i):loss[i],'Var'+str(i):var[i],'Best'+str(i):best[i]})
            
            #add information of this optimization for this d,n0
            runstats = runstats.append(pd.DataFrame(solveInfoShort,index=[n0])) #add this info
            
            if (return_option == 'frac_sol'):
                list_X.append([n0,d,X.value])

    if(verbose==2):
        print(runstats)
        
    if(verbose==1):
        print("The total number of cases tested is:")
        print(len(runstats))
        print("The number of cases where the rank is exact is:")
        print(len(runstats[runstats['d']==runstats['rank']]))
        
    if(save):
        runstats.to_csv(savedPath,index=False)
    
    if (return_option == 'frac_sol'):
        return [runstats,list_X]
    
    return runstats;