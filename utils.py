# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:05:09 2014

@author: zah
"""
import numpy as np
from sympy.printing.lambdarepr import LambdaPrinter
import numba

#more nonsensical code...
@numba.jit('void(f8[:],f8[:],u2)', nopython=True)
def memcopy(dest, src, size):
    """Copy the first `size` elements of `src` into the first `size` 
    elements of `dest`. 
    Note that there is no bounds checking and can break the program,"""
    
    for i in range(size):
        dest[i] = src[i]

def reps_to_converge(x, value = 0.1):
    return np.argwhere(x<value)[0][0] + 1

class SubstitutionPrinter(LambdaPrinter):
    d = {}
    def _print_Symbol(self, expr):
        return self.d.get(expr, super()._print_Symbol(expr) )

class ArraySubstitutionPrinter(SubstitutionPrinter):
    def __init__(self, arr_name ,arr_symbols,*args, **kwargs):
        self.d = {symbol : '%s[%i]'%(arr_name, i) 
                 for i, symbol in enumerate(arr_symbols)}
        super().__init__(*args, **kwargs)

class ReplaceComaPrinter(LambdaPrinter):
    def _print_Symbol(self, expr):
        result = super()._print_Symbol(expr).replace(',' , '_')
        return result

class NeuralPrinter(ArraySubstitutionPrinter, ReplaceComaPrinter):
    pass



def cv_split(replica, prob_testing = 0.3, even_splits = None):
    if even_splits is not None:
        raise NotImplementedError
    is_validation = np.random.rand(*replica.shape) < prob_testing
    test = replica[is_validation]
    train = replica[~is_validation]
    return train, test