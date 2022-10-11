import numpy as np
import sympy as sp

x = sympy.Symbol('x')
y = sympy.Symbol('y')

b = sympy.Symbol('b')

cost = b * np.log(np.exp(x/b) + np.exp(y/b))

dcdx = cost.diff(x)

dcdy = cost.diff(y)