'''
This is the script for testing class ROBPCA script from ENGO629-ROBPCA Copyright (c) 2015 Jeremy Steward
Author: Rong Jin
Date: 08/30/2019
'''
import numpy as np
import sys
from numpy import linalg as LA
sys.path.insert(1, '...\ENGO629-ROBPCA-master') #path where you put folder

from engo629.robust_pca import ROBPCA

'''
Load dataset
'''
X = np.genfromtxt("...\ENGO629-ROBPCA-master\iris.data", delimiter=',') 
X = X.T

'''
Convert dataset to array
'''
X = np.array(X)[:,:-1] #iris, 150*4
print("Original X:\n", X)
print("dim of original X: ", X.shape)

'''
Compute bias value for dataset
'''
# ~ n,p = X.shape 
# ~ fro = LA.norm(X, 'fro')
# ~ bv = np.sqrt(np.square(fro)/n)
# ~ print("fro norm= ", fro)
# ~ print("n= ", n)
# ~ print("biasvalue= ", bv)

'''
Add bias into original uncentered dataset as an additional column
'''
# ~ bias = np.zeros([len(X),1]) + bv
# ~ X_bias = np.append(X, bias, 1)
# ~ print("X uncentered with bias:\n", X_bias)
# ~ print("dim of X_bias: ", X_bias.shape)


'''
Romove outliers data points to compute projection error
'''
# ~ #iris 
# ~ X_rmbias = np.delete(X, (30,51,61,78,82,97,127,138), axis = 0)
# ~ print("new X:\n", X_rmbias)
# ~ print(X_rmbias.shape)

# ~ # 7points dataset
# ~ X_rmbias = np.delete(X, (4,5), axis = 0)
# ~ print("X of removing bias:\n", X_rmbias)
# ~ print(X_rmbias.shape)



if __name__ == '__main__':
  '''
    compute error
  '''
  # ~ n, p = X_rmbias.shape
  # ~ e = ROBPCA(X_rmbias)
  # ~ Z, PC, centre, newX = e.reduce_to_affine_subspace(X_rmbias)
  # ~ print("Z = \n", Z)
  # ~ print("dim of Z: ", Z.shape)
  # ~ print("PC = \n", PC)
  # ~ print("mean = ", centre)
  # ~ print("centering X:\n", newX)
  # ~ newM = PC@PC.T@newX.T
  # ~ dist = 0
  # ~ for j in range(n):
	  # ~ for i in range(p):
		  # ~ dist += np.square(newX.T[i][j] - newM[i][j])
  # ~ print("dist=", dist)
  # ~ error = dist / n
  # ~ print("error= ", error)

  ax = ROBPCA(X)
  # ~ Z, PC = ax.reduce_to_affine_subspace(X)
  # ~ print("Z = \n", Z)
  # ~ print("dim of Z: ", Z.shape)
  # ~ print("PC = \n", PC)
  
  # ~ #uncentered + bias
  # ~ axb = ROBPCA(X_bias)
  # ~ Zb, PCb = axb.reduce_to_affine_subspace(X_bias)
  # ~ print("Zb = \n", Zb)
  # ~ print("dim of Zb: ", Zb.shape)
  # ~ print("PCb = \n", PCb)
  
  #number of outliers for original
  hx = ax.num_least_outlying_points()
  print("# of hx with least outlyingness= ", hx)
  
  # ~ #number of outliers for original uncentered + bias
  # ~ hxb = axb.num_least_outlying_points()
  # ~ print("# of hxb with least outlyingness= ", hxb)
  
  #number of outliers for centered + no bias
  #Z is new matrix mean-centered from X
  # ~ az = ROBPCA(Z)
  # ~ hz = az.num_least_outlying_points()
  # ~ print("# of hz with least outlyingness= ", hz)

  # ~ dv = a.direction_through_hyperplane(Z)
  # ~ print("direction vector between two points in Z:\n", dv)
  
  H0 = ax.find_least_outlying_points(X)
  print("indices of  h observations of data points from X which have the least outlyingness:\n", H0)
  
  # ~ H0z = az.find_least_outlying_points(Z)
  # ~ print("indices of  h observations of data points from centering X which have the least outlyingness:\n", H0z)
  
  # ~ H0b = axb.find_least_outlying_points(X_bias)
  # ~ print("indices of  h observations of data points from X + bias which have the least outlyingness:\n", H0b)
  
  
  # ~ print(ax.compute_pc())
