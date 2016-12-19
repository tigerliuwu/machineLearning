
# In[3]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# # case 1: one variable
# ## load data from local file to numpy

# In[4]:

# using numpy to load data as Array
datafile = 'ex1/data/ex1data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True)


# In[5]:

print np.shape(cols)
X = cols[:-1].T # X is the samples data
Y = cols[-1:].T # Y is the samples label


# Out[5]:

#     (2, 97)
# 

# In[6]:

plt.figure(figsize=(10,6))
plt.plot(X,Y,'r+',label='sample')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()


# Out[6]:

# image file:

# In[7]:

iteration = 150 
alpha = 0.01 #learning rate = 0.01


# In[8]:

'''
X:
'''
def h(X, theta):
    return np.dot(X, theta)

def costFunction(X,Y, theta):
    A = h(X, theta) - Y
    m = Y.size
    return float(1.0/(2*m)) * np.sum(np.dot(A.T,A))


# In[9]:

X = np.insert(X,0,1,axis=1)
init_theta = np.zeros((X.shape[1],1))
print costFunction(X,Y,init_theta)


# Out[9]:

#     32.0727338775
# 

# In[10]:

def gradientDescent(X, Y, theta, iteration = 50, alpha = 1.0):
    hist_theta = [] # used to store the theta for every iteration
    ce = [] # cost error
    for it in xrange(iteration):
        hist_theta.append(list(theta[:,0])) # save the current theta
        ce.append(costFunction(X,Y, theta))
        theta = theta - alpha/len(Y) * np.dot(X.T,(h(X,theta) - Y)) 
    return hist_theta, ce, theta

hist_theta, ce, theta = gradientDescent(X, Y, init_theta, iteration=1500, alpha=0.01)


# In[11]:

print theta


# Out[11]:

#     [[-3.63029144]
#      [ 1.16636235]]
# 


# In[12]:

plt.figure(figsize=(10,6))
plt.plot(X[:,1],Y[:,0],'r+',label='training data')
plt.xlabel('iteration')
plt.ylabel('cost function')
plt.plot(X[:,1],h(X,theta),'y-',markersize=10,label='final model')
plt.legend()


# Out[12]:

#     <matplotlib.legend.Legend at 0x7f821eb58210>

# image file:

# In[39]:

def plotConvergence(ce):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(ce)), ce, 'bo', markersize=10, label='cost function')
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.xlim((-0.05*len(ce),1.05*len(ce)))
    plt.ylim((4,7))
    plt.legend()
    plt.show()

plotConvergence(ce)


# Out[39]:

# image file:

# # case2 : multiple variables

# In[14]:

# using numpy to load data as Array
datafile = 'ex1/data/ex1data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)


# In[26]:

X = cols[:-1].T
Y = cols[-1:].T
X = np.insert(X, 0, 1, axis=1) # insert 1 to the column index = 0

plt.figure(figsize=(10,8))
plt.xlim((-100,5000))
plt.ylim((-1,11))
plt.grid(True)
plt.hist(X[:,0],label='col1')
plt.hist(X[:,1],label='col2')
plt.hist(X[:,2],label='col3')
plt.title('clearly we need feature normalization')
plt.xlabel('column value')
plt.ylabel('column count')
plt.legend()
plt.show()


# Out[26]:

# image file:

# As we can see above, we definetly need feature scaling.

# In[27]:

# 
def featureScaling(X):
    normX = np.empty(X.shape)
    meanX = []
    devX = []
    for dex in xrange(X.shape[1]):
        meanX.append(np.mean(X[:,dex]))
        devX.append(np.std(X[:,dex]))
        if not dex:
            normX[:,dex] = 1
        else:
            normX[:,dex] = (X[:,dex] - meanX[-1])/devX[-1]
    
    return normX, meanX, devX

normX, meanX, devX = featureScaling(X)
        
        


# In[36]:

plt.figure(figsize=(10,8))
#plt.xlim((-100,5000))
#plt.ylim((-1,11))
plt.grid(True)
plt.hist(normX[:,0],label='col1')
plt.hist(normX[:,1],label='col2')
plt.hist(normX[:,2],label='col3')
plt.title('clearly we need feature normalization')
plt.xlabel('column value')
plt.ylabel('column count')
plt.legend()
plt.show()


# Out[36]:

# image file:

# In[38]:




# Out[38]:

#     (47, 3)

# In[ ]:



