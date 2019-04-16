import numpy as np
import math as m
from scipy.sparse import csc_matrix, vstack
from cvxopt import matrix, solvers

class PolyOpt:
    
    def __init__(self):
        self.n = 0
        self.order = 0
        self.degree = 0
        self.D = 0
        self.tnm = 0
        
        self.c = None
        self.G = None
        self.h = None
        self.A = None
        self.b = None   
        self.dims = {'l': 0,'q': [],'s': []}
        
        self.binom = None
        self.power = None
        self.base = None
        
        self.ineqcons = []
        self.eqcons = []
        
#   Create the binomial index array used to locate variables from their powers
#   Outputs: 
#   binom:      the binomial index array
    @staticmethod
    def GenBinom(n,D):
    	binom = np.zeros((n,D+2))
    	binom[0] = range(0,D+2)
    	for i in range(1,n):
    		binom[i] = np.cumsum(binom[i-1])
    	return binom.astype(int)
        
#   Generate the power index set matrix in a recursive manner
#   Inputs: 
#   nig:        number of columns in current construction
#   cur:        current value for all elements in row to sum to
#   Outputs: 
#   v:          a matrix whose rows are all vectors with ndig digits summing 
#               up to current value cur
    @classmethod
    def GenPower(cls,ndig,cur):
#       Only continue if number of columns is greater than one
        if ndig > 1:
#           Initialise current row
            v = np.zeros((1,ndig))
#           Only continue if the current value is greater than zero
            if cur > 0:
                r = 0
                for k in range(cur,-1,-1):
#                   Recursive call
                    w = cls.GenPower(ndig-1,cur-k)
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
        return v.astype(int)
    
#   Create the matrix of powers used to construct the moment and localisation
#   matrices
#   Outputs: 
#   base:       3D matrix of powers 
    def GenBase(self):
#       Determine size of matrix
        dmm = self.binom[self.n-1,self.D/2+1]
        base = np.zeros((dmm,dmm,self.n))
#       Initialise the first column of the matrix
        base[:,0,:] = self.power[0:dmm,:]
        for j in range(1,dmm):
            for k in range(0,self.n):
                base[:,j,k] = base[j,0,k]+base[:,0,k]
        return base.astype(int)
    
#   Generate all the index arrays and store them as integer values   
    def GenInd(self):
#       Generate index arrays
        self.binom = self.GenBinom(self.n,self.D)
        self.power = self.GenPower(self.n+1,self.D)[:,1:]
        self.base = self.GenBase()
        return None

#   Converts a 3D matrix of powers into a 2D matrix of indices   
#   Inputs: 
#   mpow:       3D matrix of powers 
#   Outputs: 
#   mind:       2D matrix of indices 
    def Pow2Ind(self,mpow):
#       Extract number of variables from size of mpow
        nvar = np.size(mpow,2)
#       Initialise matrix of indices
        mind = np.ones((1,np.size(mpow,0)*np.size(mpow,1)))
        for k in range(nvar):
            mind += self.binom[nvar-1-k,np.sum(mpow[:,:,k:nvar],axis=2).flatten()]
        mind = mind.reshape((np.size(mpow,0),np.size(mpow,1)))
        return mind.astype(int)
 
#   Generate the SDP constraint matrix with values corresponding to the 
#   moment matrix of this problem
    def SDPCon(self):
#       Define the dimensions of the moment matrix
        nmm = self.binom[-1,self.order+1]
        nmm2 = nmm**2
        mpow = self.base
#       Generate the column index array
        cind = self.Pow2Ind(mpow).reshape(nmm2)
#       Build G matrix in compressed column storage using row and column indices
        self.G = csc_matrix((np.ones((nmm2)), (range(nmm2), cind)), shape=(nmm2, self.tnm+1))
#       Update the dimensions of the cone dictionary
        self.dims['s'] = [nmm]
        return None

#   Update the SDP constraint matrix with localisation matrix entries 
#   containing inequality constraint information
#   Inputs:
#   g:      vector of coefficients describing the polynomial constraint (look
#           at comments for function "sdpr" for information regarding the 
#           lexicographic ordering of this coefficient vector)
#           Further note, g must be written as <= 0, if >= required, multiply
#           by -1
#   degree: degree of the polynomial
    def InequalityCon(self,g,degree):
#       Find indices of nonzero elements in coefficient vector
        ind = np.nonzero(g)[0]
#       Find nonzero coefficients
        cp = g[ind]
#       Find powers for these monomial terms
        ppow = self.power[ind]
#       Number of nonzero monomial terms
        nmon = len(ind)
        ppow = ppow.reshape((nmon,1,self.n))
#       Determine the dimensions of the localisation matrix
        nlm = self.binom[-1,1+(self.order-int(np.ceil(float(degree)/2.)))]
        nlm2 = nlm**2
        mpow = self.base[0:nlm,0:nlm,:]
        mpow = mpow.reshape((1,nlm2,self.n))
#       Localise the index information
        ppow = np.tile(mpow,(nmon,1,1))+np.tile(ppow,(1,nlm2,1))
#       Determine the column indices
        cind = self.Pow2Ind(ppow).flatten('F')
#       Row indices
        rind = np.tile(range(nlm2),(nmon,1)).flatten('F')
        cp = np.tile(cp,(1,nlm2))
        cp = cp.flatten('F')
#       Append the localisation matrix onto SDP constraint matrix
        self.G = vstack((self.G,csc_matrix((cp, (rind, cind)), shape=(nlm2, self.tnm+1)))).tocsr()
#       Update the dimensions of the cone dictionary
        self.dims['s'].append(nlm)
        return None
   
#   Update the SDP constraint matrix with localisation matrix entries 
#   containing inequality constraint information
#   Inputs:
#   g:      vector of coefficients describing the polynomial constraint (look
#           at comments for function "sdpr" for information regarding the 
#           lexicographic ordering of this coefficient vector)
#           Further note, g must be written as <= 0, if >= required, multiply
#           by -1
#   degree: degree of the polynomial
    def EqualityCon(self,g,degree):
#       Find indices of nonzero elements in coefficient vector
        ind = np.nonzero(g)[0]        
#       Find nonzero coefficients
        cp = g[ind]
#       Find powers for these monomial terms
        ppow = self.power[ind]
#       Number of nonzero monomial terms
        nmon = len(ind) 
        ppow = ppow.reshape((nmon,1,self.n))
#       Determine the dimensions of the matrix
        nlm = self.binom[-1,1+2*self.order-degree] 
        mpow = self.power[0:nlm,:]
        mpow = mpow.reshape((1,nlm,self.n))
#       Localise the index information
        ppow = np.tile(mpow,(nmon,1,1))+np.tile(ppow,(1,nlm,1)) 
#       Determine the column indices
        cind = self.Pow2Ind(ppow).flatten('F')
#       Row indices
        rind = np.tile(range(nlm),(nmon,1)).flatten('F')
        cp = np.tile(cp,(1,nlm))
        cp = cp.flatten('F')
#       Append the localisation matrix onto SDP constraint matrix
        self.A = vstack((self.A,csc_matrix((cp, (rind, cind)), shape=(nlm, self.tnm+1)))).tocsr()
        return None
      
#   Take objective function coefficients and create the linear objective 
#   function for the conic LP
#   Inputs:
#   f:      vector of coefficents for objective function
    def ConicObj(self,f):
#       Initialise the linear objective to be of size corresponding the total
#       number of moments in this problem
        obj = np.zeros(self.tnm)
#       Find indices of nonzero elements in coefficient vector
        ind  = np.nonzero(f[0])[0]
#       Find nonzero coefficients
        cp = f[0][ind]
#       Find powers for these monomial terms
        ppow = self.power[ind]
#       Number of nonzero monomial terms
        nmon = len(ind)
        ppow = ppow.reshape((nmon,1,self.n))
        cind = self.Pow2Ind(ppow)-1
        obj[cind] = cp
        self.c = obj.reshape((self.tnm,1))
        return None
        
#   Performs the SDP relaxation method for the unconstrained polynomial
#   optimisation problem
#   Inputs: 
#   obj:    list containing objective function information, including a vector
#           of coefficients (elements in lexicographic
#           order matching the power index matrix e.g. for a polynomial 
#           p(x1,x2) = 1 + 2x1^2 + 5x1x2^2, phat = [1 0 0 2 0 0 0 0 5 0]^T)
#           and the degree of the polynomial
#   n:      number of variables
#   cons:   (optional) list containing inequality constraint information, 
#           including a vector of coefficients (in same ordering as before) 
#           and degree
#           NOTE: cons is a nested list with each element of the outer list
#           containing information about each objective 
#   order:  (optional) the relaxation order of the problem
#   Outputs: 
#   c,G,h:  vectors and matrices for the linear conic LP problem 
#           min c^Tx s.t. Gx+s=h, s>=0, where s is a cone 
#   dims:   dictionary field defining the structure of the cone s
    def ConstructSDP(self,obj,n,order=None,ineqcons=[],eqcons=[]):
#       Initialise important variables
        self.n = n
        degrees = [obj[1]]
        for con in ineqcons:
            degrees.append(con[1])
        self.degree = max(degrees)
        if order == None:
            self.order = int(np.ceil(float(self.degree)/2.))
        else:
            self.order = order
        self.D = 2*self.order
#       Generate the index arrays
        self.GenInd()
#       Define the total number of moments tnm
        self.tnm = self.binom[-1,-1]
        self.ConicObj(obj)
#       Build the SDP constraint matrix for the moment matrix
        self.SDPCon()
#       Append the SDP constraints arising from inequality constraints to the
#       SDP constraint matrix
        for con in ineqcons:
            self.InequalityCon(con[0],con[1])
        for con in eqcons:
            self.EqualityCon(con[0],con[1])
#       Define the vectors and matrices for the linear conic LP problem 
        c = matrix(self.c[1:])
        G = matrix(-self.G[:,2:].toarray())
        h = matrix(self.G[:,1].toarray())
        dims = self.dims
        if self.A is not None:
            A = matrix(-self.A[:,2:].toarray())
            b = matrix(self.A[:,1].toarray())
        else:
            A = None
            b = None  
        return c,G,h,A,b,dims
    
    def AddLinearConstraints(self,A,b_l,b_u):
        M = np.block([[b_u.reshape(-1,1), -A],[-b_l.reshape(-1,1), A]])
        for i in range(M.shape[0]):
            self.ineqcons.append([M[i,:],1])
        return None
    
    def AddNonLinearConstraint(self,coeff,deg,b_l,b_u):
        if not np.isinf(b_l):
            gbl = np.zeros_like(coeff)
            gbl[0] = -b_l
            gbl += coeff
            conl = [gbl,deg]
            self.ineqcons.append(conl)
        if not np.isinf(b_u):
            gbl = np.zeros_like(coeff)
            gbl[0] = b_u
            gbl -= coeff
            conl = [gbl,deg]
            self.ineqcons.append(conl)
        return None
    
    def Optimise(self,obj,n,order=None):
        c,G,h,A,b,dims = self.ConstructSDP(obj,n,order,self.ineqcons,self.eqcons)
        solvers.options['show_progress'] = False
        sol = solvers.conelp(c, G, h, dims, A, b)
        solution = {'status': None, 'obj': None, 'iterations': None, 'x': None}
        xopt = np.array(sol['x'][:n])
        solution['iterations'] = sol['iterations']
        if sol['status'] == 'optimal':
            solution['status'] = sol['status']
            solution['x'] = np.array(xopt)
            solution['obj'] = sol['primal objective'] + obj[0][0,0]
        return solution

    