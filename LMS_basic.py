import scipy as sp
import numpy as np
from nptyping import Complex128
import matplotlib.pylab as plt

G_train=sp.io.loadmat(r'', appendmat = True)['G_train']

data_input = sp.io.loadmat(r'', appendmat = True)['Sin']

y_output=sp.io.loadmat(r'', appendmat = True)['Sout']


data_input = np.array(data_input, dtype = np.complex128)[0,:]

y_output=np.array(y_output,dtype=Complex128)[0,:]

G_train = np.array(G_train, dtype = np.complex128)

w=np.zeros(1365,dtype=Complex128)

data_input= data_input[:50000]

y_output=y_output[:50000]

mse1=np.sum(data_input*np.conj(data_input))

mu=0.00044#2/(mse1*1365)

n=data_input.shape(0)

for i in range(0,n):

  est=np.einsum('i,i',np.conj(w),G_train[i])

  error=y_output[i]-est

  w=w+mu*G_train[i]*np.conjugate(error)



Y=np.einsum('ij,j->i',G_train,w)

plt.plot(abs(data_input),abs(Y),'g.')

plt.show()
