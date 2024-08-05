import numpy as np
from nptyping import Complex128
import scipy as sp
import matplotlib.pyplot as plt

x = sp.io.loadmat(r'E:/Work table/OMP/GML/data/data_for_OMP', appendmat = True)['G_train']
y= sp.io.loadmat(r'E:/Work table/OMP/GML/data/data_for_OMP', appendmat = True)['data_output_train']

x = np.array(x, dtype = Complex128)
y = np.array(y, dtype = Complex128)






def error_sq(y,y_t):
   #for i in range(0,np.size(y,1)):
     # y[:,i]=y[:,i]*mas[:,i]
   er=np.linalg.norm(y-np.transpose(y_t))**2
   return er

#BIC
# M number of max index y
# n number of iteration

def BIC(in_y,in_y_tilda,M,n): 

   bic=2*M*np.log(error_sq(in_y,in_y_tilda))+n*2*np.log(2*M)
   return bic


#Find number of max collumn correlation matrix
     

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
bic=np.array((1,n),dtype=float)
s=0
x_col=np.matrix.getH((x[s]))/np.linalg.norm(x[s])


#OMP main algorithm

for iteration in range(0,n):    
    g=np.dot(x_col.reshape(m,1),r)
    s=max_regeressor(g)
    mask[0][s]=True
    X[s]=x[s]
    pinv=np.linalg.pinv(X)
    h=np.dot(pinv,y.T)
    y_tilda=np.dot(X,h)
    r=y-np.transpose(y_tilda)
    bic[iteration]=BIC(y,y_tilda,n,m)
    print(iteration+1,bic[iteration])

iter=np.arange(1,n,dtype=int)
plt.plot(iter,bic)
plt.show()
    
    






 