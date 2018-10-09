'''
Container for activation functions.

Largely copied from neat-python. Copyright 2015-2017, CodeReclaimers, LLC.
'''
import math
import types
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid_activation(x):
    '''
    Hard sigmoid function [-1, 1]
    '''
    return 1 / (1+math.exp(-x))
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))

def tanh_activation(x):
    '''
    Hard hyperbolic tanger function [-1, 1]
    '''
    return math.tanh(x)
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)

def sin_activation(x):
    '''
    Hard sin function [-1, 1]
    '''
    return math.sin(x)
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)

def tan_activation(z):
    return math.tan(z)

def cos_activation(z):
    return math.cos(z)

def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z**2)

def dhn_gauss_activation(x):
    mu = 0
    z = abs(x)
    return math.exp(-100.0 * (x-mu)**2)

def dhn_gauss_activation_2(x):
    mu = 2
    return math.exp(-100.0 * (x-mu)**2)

def relu_activation(x):
    return x if x > 0.0 else 0.0

def log_activation(x):
    z = max(1e-7, x)
    return math.log(x)

def exp_activation(x):
    x = max(-60.0, min(60.0, z))
    return math.exp(x)

def linear_activation(x):
    return x

class InvalidActivationFunction(TypeError):
    pass

def validate_activation(function):
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")
    if function.__code__.co_argcount != 1: # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")

class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """
    def __init__(self):
        self.functions = {}
        self.add('sigmoid', sigmoid_activation)
        self.add('sin', sin_activation)
        self.add('cos', cos_activation)
        self.add('relu', relu_activation)
        self.add('linear', linear_activation)
        self.add('gauss', gauss_activation)
        self.add('dhngauss', dhn_gauss_activation)
        self.add('dhngauss2', dhn_gauss_activation_2)

    def add(self, name, function):
        validate_activation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))
        return f

    def is_valid(self, name):
        return name in self.functions