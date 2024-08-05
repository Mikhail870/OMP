import numpy as np
from nptyping import Complex128
import scipy as sp

x = sp.io.loadmat(r'E:/Work table/OMP/GML/data/data_for_OMP', appendmat = True)['G_train']
y= sp.io.loadmat(r'E:/Work table/OMP/GML/data/data_for_OMP', appendmat = True)['data_output_train']

x = np.array(x, dtype = Complex128)
y = np.array(y, dtype = Complex128)

#Find number of max collumn correlation   
def max_regeressor(a): 
   n=np.size(a,axis=1)
   max=np.zeros((1,n))
   for i in range(0,n):
     max[0][i]=np.linalg.norm(a[:,i])
   return np.argmax(max)


#inicialization variable and matrix
m=np.size(x,1)
n=np.size(x,0)
r=np.array(y,dtype=Complex128)
mask=np.zeros((1,n),dtype=int)
X=np.zeros((n,m),dtype=Complex128)
s=0
z=x

#OMP main algorithm
for iteration in range(0,n):   
    Z=np.matrix.getH((z[s]))/np.linalg.norm(z[s]) 
    g=np.dot(np.reshape(Z,(m,1)),r)
    s=max_regeressor(g)
    mask[0][s]=True
    X[s]=x[s]
    p=np.dot(z,np.matrix.getH(np.reshape(z[s],(1,m))))
    z=z-np.kron(p,np.reshape(z[s],(1,m)))
    pinv=np.linalg.pinv(X)
    h=np.dot(pinv,y.T)
    y_tilda=np.dot(X,h)
    r=y-np.transpose(y_tilda)
    print(np.linalg.norm(r))
    
    