
# **the logistic Regression can only help resolve the problem of binary classification, what can we do when there are multiple different output labels?**
# 
# There are 2 solutions:<br/>
#  - **one-vs-all**
#  - **neural network**
# 
# # load the data and visualization
# 
# ## import the needed modules

# In[48]:

#show the figures buildin the notebook
# get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
from scipy import optimize
import matplotlib.cm as cm #Used to display images in a specific colormap
import random


# ## load the data from the file as matrix

# In[49]:

fileName='ex3/data/ex3data1.mat'
samples = scipy.io.loadmat(fileName)


# In[50]:

# extract to get (X,y) samples
X = samples['X']
X = np.insert(X,0,1,axis=1)  # insert 1 before index 0 as a column
Y=samples['y']
# since the handy digit 0 is labeled as 10
Y=np.array([i % 10 for i in Y])

print "X's shape is: %s,X[0]'s shape is: %s" % (X.shape, X[0:1].shape)
print "y's shape is: %s, unique elements in y: %s" % (Y.shape, np.unique(Y))


# Out[50]:

#     X's shape is: (5000, 401),X[0]'s shape is: (1, 401)
#     y's shape is: (5000, 1), unique elements in y: [0 1 2 3 4 5 6 7 8 9]
# 

# ## visualization the samples

# In[51]:

def myPlotImg(myX):
    
    
    xPixes, yPixes = (20,20)
    nrows,ncols = (10,10)
    pad = 1
    
    # this variable is used to store all the 100 samples data which are used to visualize
    big_pic = np.zeros((nrows * (xPixes + pad),ncols * (yPixes + pad)))
    
    cr,cl=(0,0) #cr stands for rows, cl stands for column
    for i in xrange(nrows):
        for j in xrange(ncols):
            # extract an sample as a 20 * 20 matrix
            sample = myX[random.randint(1,X.shape[0])-1] #randomly get a sample
            sample = sample[1:].reshape(xPixes,yPixes) # ignore the first column with value 1
            
            big_pic[i * xPixes+pad:(i+1)*xPixes+pad,j*yPixes+pad:(j+1)*yPixes+pad] = sample.T
    
    plt.figure(figsize=(6,6))
    img = scipy.misc.toimage(big_pic)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.imshow(img,cmap=cm.Greys_r)
    plt.show()

myPlotImg(X)   


# Out[51]:

# image file:

# # one-vs-all
# **For this solution, we make good use of logistic regression to complete this.**
# 
# **Mechnism**
# 
# We take one label as the expected one(y=1) and the other labels are the negative labels(y=0)
# 
# ## implementation
# 
# ### implement cost and gradient function

# In[65]:

# define the sigmoid function:
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# define the cost function with regularization
def costFun(theta, myX, myY, mylambda = 0.0):
    theta = theta.reshape(-1,1)
    
    z = np.dot(X,theta)
    T = sigmoid(z)
    if np.isnan(T[0][0]):
        print "there is nan value!!!!!!!!" 
        
    m = len(X)
    result = np.sum(myY * np.log(T) + (1 - myY)*np.log(1-T)) * (-1) / m
    theta2 = theta[1:,:]
    result = result + (mylambda / (2* m)) * np.dot(theta2.T,theta2)
    if np.isnan(result):
        return np.inf
    return result
    

# define the gradient function of cost function
def gradientFun(theta, myX, myY, mylambda =0.0):
    theta = theta.reshape(-1,1)
   # print theta, theta.shape
    Z = np.dot(myX, theta)   # m*1
    T = sigmoid(Z)
    m = len(myX)
    result = np.dot(myX.T,T - myY) / m  # n * 1
    theta[0] = 0
    result = result + (mylambda / m) * theta
    return result.flatten()

def oneVSAll(myX, myY, mylambda=0.0):
    # get all the possible labels
    labels = np.unique(myY)
    
    theta = np.empty((len(labels), myX.shape[1])) # labels * (n+1)
    for xclass in labels:
        ## one VS All(0)
        print "optimizing for the handwriting number:%d" % xclass
        tY = np.array([1 if v==xclass else 0 for v in myY.flatten()]).reshape(-1,1) # m * 1
        res = optimize.minimize(costFun,np.zeros((myX.shape[1],1)), args=(myX,tY,mylambda),method='BFGS',
                                 jac=gradientFun, options={'maxiter':100,'disp':True})
        theta[xclass] = res.x
    
    print "one vs all Done"
    
    return theta
        
theta = oneVSAll(X,Y,mylambda=0.1)    


# Out[65]:

#     optimizing for the handwriting number:0
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.011179
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     optimizing for the handwriting number:1
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.016092
#              Iterations: 100
#              Function evaluations: 103
#              Gradient evaluations: 103
#     optimizing for the handwriting number:2
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.057727
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     optimizing for the handwriting number:3
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.063839
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     optimizing for the handwriting number:4
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.039164
#              Iterations: 100
#              Function evaluations: 105
#              Gradient evaluations: 105
#     optimizing for the handwriting number:5
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.062603
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     optimizing for the handwriting number:6
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.023639
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     optimizing for the handwriting number:7
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.036296
#              Iterations: 100
#              Function evaluations: 103
#              Gradient evaluations: 103
#     optimizing for the handwriting number:8
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.084091
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     optimizing for the handwriting number:9
#     Warning: Maximum number of iterations has been exceeded.
#              Current function value: 0.077913
#              Iterations: 100
#              Function evaluations: 102
#              Gradient evaluations: 102
#     one vs all Done
# 

# ### predict the samples' accuracy

# In[70]:

# define the predict function to return the predict number array
def predictOneVSAll(all_theta,myX):
    z = np.dot(myX,all_theta.T) # m * num_labels
    t = sigmoid(z)
    result = np.argmax(t,axis=1).reshape(-1,1)
    return result

predict = predictOneVSAll(theta, X)
precision = float(np.sum(predict ==Y)) / len(Y)  * 100
print "training set accuracy:%0.2f" % precision


# Out[70]:

#     training set accuracy:94.96
# 
