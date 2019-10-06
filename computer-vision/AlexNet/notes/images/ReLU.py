import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# functions to plot
x = np.arange(-2.5, 2.5, 0.01)
tanhx = np.tanh(x)
exp_funct = (1 + np.exp(-x))**-1
max_x = np.maximum(0,x)

fig, ax = plt.subplots()
ax.plot(x, tanhx,'r', label="$f(x) = tanh(x)$")
ax.plot(x, exp_funct,'g', label="$f(x) = (1 + e^{-x})^{-1}$")
ax.plot(x, max_x,'b', label="$f(x) = max(0, x)$")

ax.grid()
ax.set_title("Graphs of Activation Functions")
plt.legend(loc='lower right')

# add cartesian axes
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')

# set x-,y- axes labels
fig.text(0.5, 0.04, 'x', ha='right', va='center')
fig.text(0.06, 0.5, 'y', ha='center', va='bottom', rotation='vertical')

fig.savefig("ReLU.png")
plt.show()
