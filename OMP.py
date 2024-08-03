import numpy as np
from nptyping import Complex128
import scipy as sp

x = sp.io.loadmat(r'', appendmat = True)['G_train']
y= sp.io.loadmat(r'', appendmat = True)['data_output_train']

x = np.matrix(x, dtype = Complex128)
y = np.matrix(y, dtype = Complex128)

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
r=np.matrix(y,dtype=Complex128)
mask=np.zeros((1,n),dtype=int)
X=np.matrix(np.zeros((n,m),dtype=Complex128))
s=0
x_col=np.matrix.getH((x[s]))/np.linalg.norm(x[s])

#OMP main algorithm
for iteration in range(0,n):    
    g=np.matmul(x_col,r)
    s=max_regeressor(g)
    mask[0][s]=1
    X[s]=x[s]
    pinv=np.linalg.pinv(X)
    h=np.matmul(pinv,y.T)
    y_tilda=np.matmul(X,h)
    r=y-np.transpose(y_tilda)
    print(np.linalg.norm(r))
    