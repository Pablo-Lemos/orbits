from pysr import PySRRegressor
import numpy as np
from sympy import *
import os

X = 2 * np.random.randn(100, 4)
Y = 3 * X[:, 0] * np.sin(X[:, 2]) + 2 * X[:, 3]**2 + 1

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=50,
    equation_file= os.path.join("./data/saved_equations/", 'SR Example'),   # Save equations in a file
    binary_operators=["+", "*"],    #  allowed binary operators
    unary_operators=[       # allowed unary operators
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
	    # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X, Y)

print(model)
print(model.sympy())
