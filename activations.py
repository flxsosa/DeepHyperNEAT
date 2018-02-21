"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""
from __future__ import division
import math
import types


def sigmoid_activation(z):
    '''
    Hard sigmoid function [-1, 1]
    '''
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    '''
    Hard hyperbolic tanger function [-1, 1]
    '''
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def sin_activation(z):
    '''
    Hard sin function [-1, 1]
    '''
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z**2)

def dhn_gauss_activation(x):
    '''
    Gaussian function as defined in DHN paper.
    '''
    # print("Input to DHN Gauss:", x)
    sigma = 1/(math.sqrt(2*math.pi))
    mu = 0
    y = (1.0/(sigma * math.sqrt(2*math.pi)))
    z = y * math.e**(-1*(x-mu)**2/(2*sigma)**2)
    # print("Output from DHN Gauss:", z)
    return z

def dhn_gauss_activation_2(x):
    '''
    Gaussian function for DHN that checks for identity.
    '''
    # print("Input to DHN Guass 2:", x)
    # sigma = 0.1
    mu = 2
    # y = (1.0/(sigma * math.sqrt(2*math.pi)))
    # z = y * math.e**(-1*(x-mu)**2/(2*sigma)**2)
    sigma = 0.01
    z = math.e**(-1*(x-mu)**2/(2*sigma)**2)
    # print("Output from DHN Gauss 2:", z/3.98)
    return z

def dhn_gauss_activation_3(x):
    '''
    Gaussian function for DHN that checks for identity.
    '''
    mu = 2
    sigma = 0.01
    z = math.e**(-1*(x-mu)**2/(2*sigma)**2)
    return z

def relu_activation(z):
    return z if z > 0.0 else 0.0


def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1 + math.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError: # handle overflows
        return 0.0
    else:
        return z

def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3

def linear(x):
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
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)
        self.add('relu', relu_activation)
        # self.add('softplus', softplus_activation)
        # self.add('identity', identity_activation)
        # self.add('clamped', clamped_activation)
        # self.add('inv', inv_activation)
        # self.add('log', log_activation)
        # self.add('exp', exp_activation)
        # self.add('abs', abs_activation)
        # self.add('hat', hat_activation)
        # self.add('square', square_activation)
        # self.add('cube', cube_activation)
        self.add('dhngauss', dhn_gauss_activation)
        self.add('linear', linear)
        self.add('dhngauss2', dhn_gauss_activation_2)
        self.add('dhngauss3', dhn_gauss_activation_3)

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
