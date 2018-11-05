"""Softmax."""

import numpy as np
import matplotlib.pyplot as plt

scores = np.array([5.0, 2.0, 6.0])#possible input

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    
    x = np.asarray(x)
    
    temp_col = np.empty((x.shape[0],))


    if len(x.shape) < 2:
        c=np.exp(x)/np.exp(x).sum()
    else:       
    #iterate through each column and perform math
        for i in x.T:
            i = np.exp(i)/np.exp(i).sum() # softmax equation
            temp_col = np.column_stack((temp_col,i)) # stack columns
        
        c = temp_col[:,1:]#remove first empty column
  
    return c
    #OR better
    #return np.exp(x)/np.sum(np.exp(x), axis = 0)


sf_scores= softmax(scores)

#print(sf_scores)

#plot softmax scores

x = np.arange(1,len(scores)+1,1);

y = softmax(scores);

# quiz for understanding what happens if inputs change in magnitude                         
y_mul = softmax(scores * 10)
y_div = softmax(scores / 10)                     

#plot the results
plt.plot(x, y,'ro',label='Softmax of Normal Input')
#plt.plot(x,y_mul,'bs',x,y_div,'g^') # quiz results
plt.show()

"""
# Plot softmax curves

x = np.arange(-2.0, 6.0, 0.1)# x-axis
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)]) #y-axis

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
"""
