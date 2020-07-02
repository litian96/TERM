import numpy as np

def std_PCA(W,d):
    '''
    Given a n x n matrix (this is equal to A^T A for an m x n data matrix A), output the top d eigenvectors of W as numpy n x d matrix.
    '''
    [eigenValues,eigenVectors] = np.linalg.eig(W)

    #sort eigenvalues and eigenvectors in decending orders
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    #take the first d vectors. Obtained the solution
    return eigenVectors[:,:d]