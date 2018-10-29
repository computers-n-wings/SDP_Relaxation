import numpy as np
from scipy.sparse import csc_matrix
from cvxopt import matrix, solvers

class sdprelaxation:
    def __init__(self):
        self.K = {'l': 0,'q': [],'s': []}
        self.As = None
        self.binom = None
        self.power = None
        self.base = None
        self.n = 0
        self.degree = 0
        self.order = 0
        self.D = 0
        self.tnm = 0
    
#   Create the binomial index array used to locate variables from their powers
#   Outputs: the binomial index array binom
    def genbinom(self):
    	binom = np.zeros((self.n,self.D+2))
    	binom[0] = range(0,self.D+2)
    	for i in range(1,self.n):
    		binom[i] = np.cumsum(binom[i-1])
    	return binom

#   Generate the power index set matrix in a recursive manner
#   Inputs: number of columns in current construction ndig and current sum value 
#   cur
#   Outputs: a matrix whose rows are all vectors with ndig digits summing up to 
#   current value cur, v
    def genpower(self,ndig,cur):
#       Only continue if number of columns is greater than one
        if ndig > 1:
#           Initialise current row
            v = np.zeros((1,ndig))
#           Only continue if the current value is greater than zero
            if cur > 0:
                r = 0
                for k in range(cur,-1,-1):
#                   Recursive call
                    w = self.genpower(ndig-1,cur-k)
                    rd = len(w)
                    v1 = np.hstack((k*np.ones((rd,1)),w))
                    r = r+rd;
                    if k == cur:
                        v = v1
                    else:
                        v = np.vstack((v,v1))
#       If number of columns is less than one, return a singleton array with 
#       value equal to cur
        else:
            v = np.array([[cur]])
        return v
 
#   Create the matrix of powers used to construct the moment and localisation
#   matrices
#   Outputs: 3D matrix of powers base
    def genbase(self):
#       Determine size of matrix
        dmm = self.binom[self.n-1,self.D/2+1]
        base = np.zeros((dmm,dmm,self.n))
#       Initialise the first column of the matrix
        base[:,0,:] = self.power[0:dmm,:]
        for j in range(1,dmm):
            for k in range(0,self.n):
                base[:,j,k] = base[j,0,k]+base[:,0,k]
        return base

#   Generate all the index arrays and store them as integer values   
    def genind(self):
#       Quit if D is not divisible by 2 (NOTE: this will not be an issue right 
#       now, but might be later)
        if self.D%2 != 0:
            ValueError("The degree must be even")
#       Generate index arrays
        self.binom = self.genbinom().astype(int)
        self.power = self.genpower(self.n+1,self.D)[:,1:].astype(int)
        self.base = self.genbase().astype(int)
        return None

#   Converts a 3D matrix of powers into a 2D matrix of indices   
#   Inputs: 3D matrix of powers mpow
#   Outputs: 2D matrix of indices mind
    def pow2ind(self,mpow):
        nvar = np.size(mpow,2)
        mind = np.ones((1,np.size(mpow,0)*np.size(mpow,1)))
        for k in range(nvar):
            mind += self.binom[nvar-1-k,np.sum(mpow[:,:,k:nvar],axis=2).flatten()]
        mind = mind.reshape((np.size(mpow,0),np.size(mpow,1)))
        return mind
 
#   Generate the SDP constraint matrices for the unconstrained polynomial 
#   optimisation problem and update the cone dictionary
    def sdpcon(self):
#       Define the dimensions of the moment matrix
        nmm = self.binom[-1,self.order+1]
        nmm2 = nmm**2
        mpow = self.base
#       Generate the column index array
        cind = self.pow2ind(mpow).reshape(nmm2)
#       Build As matrix in compressed column storage using row and column indices
        self.As = csc_matrix((np.ones((nmm2)), (range(nmm2), cind)), shape=(nmm2, self.tnm+1))
#       Update the dimensions of the PSD cone
        self.K['s'] = [nmm]
        return None

#   Performs the SDP relaxation method for the unconstrained polynomial
#   optimisation problem
#   Inputs: polynomial coefficient vector phat (elements in lexicographic
#   order matching the power index matrix e.g. for a polynomial 
#   p(x1,x2) = 1 + 2x1^2 + 5x1x2^2, phat = [1 0 0 2 0 0 0 0 5 0]^T), 
#   number of variables n, and polynomial degree 
#   Outputs: vectors and matrices for the linear conic LP problem 
#   min c^Tx s.t. Gx+s=h, s>=0, where s is a cone defined by the dictionary 
#   field dims
    def sdpr(self,phat,n,degree):
#       Initialise important variables
        self.n = n
        self.degree = degree
        self.order = int(np.ceil(float(self.degree)/2.))
        self.D = 2*self.order
#       Generate the index arrays
        self.genind()
#       Define the total number of moments tnm
        self.tnm = self.binom[-1,-1]
#       Build the SDP constraint matrix and define the cone
        self.sdpcon()
#       Define the vectors and matrices for the linear conic LP problem 
        c = phat[1:,:]
        G = -self.As[:,2:]
        h = self.As[:,1]
        dims = self.K
        
        return c,G,h,dims
    
if __name__ == '__main__':
##   Coefficient vector for 2D Styblinski–Tang function   
#    phat = csc_matrix((np.array([2.5,2.5,-8.,-8.,0.5,0.5]), (np.array([1,2,3,5,10,14]), np.zeros(6))), shape=(15, 1))
#    n = 2
#    degree = 4
    
#   Coefficient vector for 2D Rosenbrock function
    phat = csc_matrix((np.array([1.,-2.,1.,100.,-200.,100.]),(np.array([0,1,3,5,7,10]), np.zeros(6))), shape=(15, 1))
    n = 2
    degree = 4
    
    ins = sdprelaxation()
    c,G,h,dims = ins.sdpr(phat,n,degree)
    c = matrix(c.toarray())
    G = matrix(G.toarray())
    h = matrix(h.toarray())

    sol = solvers.conelp(c, G, h, dims)
    
    print sol['x'][0:n]
    
    