# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:44:19 2014

@author: zah
"""
import functools
from itertools import chain, repeat


import numpy as np
import sympy
import numba
from numba import types


from utils import NeuralPrinter, memcopy


def sigmoid_g(a):
    return 1/(1+sympy.exp(-a))

def linear_g(a):
    return a

class Node():
    pass

class InputNode(Node):
    def __init__(self, name, function=None):
        self.name = name
        self.symbol = sympy.Symbol('x_%s'%self.name)

    @property
    def formula(self):
        return self.symbol


class NetNode(Node):
    def __init__(self, name, g ,inputs = None, weights = None, theta = 0.):
        self.name = name
        if inputs is None:
            inputs = []
        self._inputs = inputs
        self.g = g
        self.theta = theta
        self.theta_symbol = sympy.Symbol('theta_%s'%name)
        self._init_weights(weights)



    def _init_weights(self, weights = None):
        inputs = self.inputs
        if weights is None:
            self.weights = np.ones(len(inputs))
        else:
            if len(weights) != len(inputs):
                raise ValueError("Weights must be the same lenght as inputs")
            self.weights = weights

        self.weight_symbols = [sympy.Symbol('w_%s__%s'%
                                 (inp.name, self.name)) for inp in inputs]

    @property
    def formula(self):
        inpart = sum([w*i.formula for w,i in
                    zip(self.weight_symbols, self.inputs)])

        a = inpart - self.theta_symbol
        return self.g(a)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value
        self._init_weights()

class HiddenNode(NetNode):
    pass

class OutputNode(NetNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ymbol = sympy.Symbol('y_%s'%self.name)

class NeuralNetwork():

    def __init__(self):
        self._init_parmams()
    
    _output_formulas = None
    _params = None
    _evaluate_sync = True
    _ev_fs = None

    @property
    def unidimentional_output(self):
        return len(list(self.output_nodes)) == 1
    
    @property
    def total_output_formulas(self):
        if self._output_formulas is None:
            self._output_fromulas = [node.formula for node in self.output_nodes]
        return self._output_fromulas

    @property
    def total_output_formula(self):
        return self.total_output_formulas[0]

    @property
    def input_symbols(self):
        yield from (node.symbol for node in self.input_nodes)

    @property
    def parameter_symbols(self):
        for node in chain(self.hidden_nodes, self.output_nodes):
            yield from node.weight_symbols
            yield node.theta_symbol

    @property
    def parameter_values(self):
        for node in chain(self.hidden_nodes, self.output_nodes):
            yield from node.weights
            yield node.theta

    def get_evaluate_functions(self):
        """Return compiled functions with fixed network parameters"""
        if self._ev_fs and self._evaluate_sync:
            return self._ev_fs
        
        ev_fs = []
        param_symbols = self.parameter_symbols
        param_values = self.parameter_values
        input_symbols = list(self.input_symbols)
        input_len = len(input_symbols)

        for f in self.total_output_formulas:
            expr = f.subs(zip(param_symbols, param_values))
            func = sympy.lambdify(input_symbols, expr)
            signature = 'float64(%s)' % ','.join(repeat('float64',input_len))

            ev_fs.append(numba.vectorize(signature)(func))
        
        self._evaluate_sync = True
        self._ev_fs = ev_fs
        return ev_fs

    def get_evaluate_function(self):
        """Return only the first function (to avoid typing zero every time)"""
        return self.get_evaluate_functions()[0]
    
    def _get_params(self):
         for node in chain(self.hidden_nodes, self.output_nodes):
             yield from node.weights
             yield node.theta
    

             
    @property
    def _node_indexes(self):
        i = 0
        for node in chain(self.hidden_nodes, self.output_nodes):
            yield i
            i += len(node.weights) + 1
    
    @property
    def parameters(self):
        return self._params
            
    
    def _init_parmams(self):
        #Initialize params
        nparams = len(list(self.parameter_symbols))
        params = np.random.normal(size=nparams)
        self._set_params(params)
    
    def reset_params(self):
        self._init_parmams()
    
    def save_params(self, file):
        np.save(file, self._params)
    
    def load_params(self, file):
        params = np.load(file)
        self._set_params(params)
    
    

    def _set_params(self, params):
        """Set network parameters where `params` is
        given in canonical order"""
        if len(params) != len(list(self.parameter_symbols)):
            raise ValueError("Incompatible parameter specification.")
        i = 0
        for node in chain(self.hidden_nodes, self.output_nodes):
            l = len(node.weights)
            node.weights = params[i:i+l]
            node.theta = params[i+l]
            i += l+1
        self._evaluate_sync = False
        self._params = params

    #TODO: make multidimentional. Now assumes scalar input
    def input_param_functions(self):
        p_fs = []
        param_symbols = list(self.parameter_symbols)
        input_symbols = list(self.input_symbols)

        for f in self.total_output_formulas:
            #TODO: multidimensional input
            signature = 'float64(float64,float64[:])'
            params = sympy.Symbol('params')
            printer = NeuralPrinter('params', param_symbols)
            func = sympy.lambdify(input_symbols + [params], f, 
                                  printer = printer, dummify = False)
            p_fs.append(numba.jit(signature, nopython=True)(func))
        return p_fs
    
    #TODO make multidimensional    
    #TODO real covariance
    def _get_chi2_func(self):
        
        func = self.input_param_functions()[0]  
        
        @numba.jit('f8(f8[:],u2,f8[:],f8[:],f8[:])', nopython = True)
        def chi2_func(params, l, X, Y, covariance):
            chi2 = 0
            for i in range(l):
                x = X[i]
                y = Y[i]
                cov = covariance[i]
                chi2 += (func(x, params) - y)**2 / cov
            return chi2
        
        return chi2_func
    
    def _get_genetic_mutate_function_orig(self):
        
        indexes = np.fromiter(self._node_indexes, dtype=np.int)
        nnodes = len(indexes)
        nparams = len(self.parameters)


        @numba.jit('void(f8[:],u2,u2,u2,f8,f8,f8[:,:,:,:],f8[:,:])', nopython = True)
        def mutate(params, rep, mutant, node, eta, chi2, node_random, iter_random):
            frm = indexes[node]
            if node == nnodes - 1:
                to = nparams
            else:
                to = indexes[node + 1]
            rite = iter_random[rep, mutant]
            const = eta/(rep + 1)**rite
            for i in range(frm, to):
                rd = node_random[rep,mutant, node, i - frm]
                params[i] += const*rd
                
                
        return mutate

    def _get_genetic_mutate_function_mutate(self):
        
        indexes = np.fromiter(self._node_indexes, dtype=np.int)
        nnodes = len(indexes)
        nparams = len(self.parameters)


        @numba.jit('void(f8[:],u2,u2,u2,f8,f8,f8[:,:,:,:],f8[:,:])', nopython = True)
        def mutate(params, rep, mutant, node, eta, chi2, node_random, iter_random):
            frm = indexes[node]
            if node == nnodes - 1:
                to = nparams
            else:
                to = indexes[node + 1]
            rite = iter_random[rep, mutant]
            const = eta*((1 + (rep+1)*np.log10(1+chi2))/(rep + 1))**rite
            for i in range(frm, to):
                rd = node_random[rep,mutant, node, i - frm]
                params[i] += const*rd
                
                
        return mutate
    
    def _random_pool(self, reps, nmutants, mutate_prob):
        indexes = np.fromiter(self._node_indexes, dtype=np.int)
        nnodes = len(indexes)
        nparams = len(self.parameters)
        max_nweights = np.max([np.max(np.diff(indexes)), nparams - indexes[-1]])
        will_mutate = np.random.rand(reps, nmutants, nnodes) < mutate_prob
        #TODO: Use this to reduce size of pool
        #total_mutants = np.sum(will_mutate)
        node_random = np.random.uniform(-1, 1,
                                size=(reps, nmutants, nnodes, max_nweights))
        
        iter_random = np.random.rand(reps, nmutants)
        
        return will_mutate,node_random,iter_random
    
    @functools.lru_cache()
    def _get_genetic_fit_function(self, target, mutate_func):


        if target == 'chi2_func':
            func = self._get_chi2_func()
        else:
            func = target()
        
        if mutate_func == 'original':
            mutate = self._get_genetic_mutate_function_orig()
        elif mutate_func == 'chiweights':
             mutate = self._get_genetic_mutate_function_chi2()
        else:
            mutate = mutate_func()
        
        nparams = len(list(self.parameter_symbols))

        nnodes = len(list(self._node_indexes))
        
        
        @numba.jit('void(f8,f8,u2,u2,f8,'
        'f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],b1[:,:,:]'
        ',f8[:,:,:,:],f8[:,:])', 
        nopython=True)
        def make_fit(reps, eta, nmutants, mutate_prob, l, params, 
                     best_params,  mutparams, X, Y, covariance, chi2,
                     will_mutate, node_random, iter_random):
                     
            starting_f = func(params, l, X, Y, covariance)
            memcopy(best_params, params, nparams)
            for rep in range(reps):
                for mutant in range(nmutants):
                    memcopy(mutparams, params, nparams)
                    for node in range(nnodes):
                        if will_mutate[rep,mutant,node]:
                            mutate(mutparams, rep, mutant, node, eta,
                                   starting_f, node_random, iter_random)

                    new_f = func(mutparams, l, X, Y, covariance)
                    
                    if new_f < starting_f:
                        memcopy(best_params, mutparams , nparams)
                        starting_f = new_f
                    chi2[rep] = starting_f
                memcopy(params, best_params, nparams)
        
        return make_fit
        
        
    def genetic_fit(self, X, Y, covariance = None, target = 'chi2_func', 
                    mutate_func = 'original',
                    reps = 10000, eta = 15, nmutants = 80, mutate_prob = 0.2):
        
        if covariance is None:
            covariance = np.ones_like(Y)
        
        params = np.array(self.parameters, dtype=np.float64)
        nparams = len(params)
        best_params = np.zeros(nparams)
        mutparams = np.zeros(nparams)
        chi2 = np.zeros(reps)
        
        
        make_fit = self._get_genetic_fit_function(target,mutate_func)
        
        l = len(X)
        will_mutate, node_random, iter_random = self._random_pool(reps, 
                                                        nmutants, mutate_prob)
        
        
        make_fit(reps, eta, nmutants, mutate_prob, l, params, best_params, 
                 mutparams, X, Y, covariance, chi2, will_mutate, node_random, 
                 iter_random)
                 
        self._set_params(params)
        return chi2
    
    def __call__(self, x):
        if self.unidimentional_output:
            return self.get_evaluate_function()(x)
        else:
            return [f(x) for f in self.get_evaluate_functions()]
    
    @property
    def input_nodes(self):
        raise NotImplementedError()


    @property
    def output_nodes(self):
        raise NotImplementedError()

    @property
    def all_nodes(self):
        raise NotImplementedError()

    @property
    def hidden_nodes(self):
        raise NotImplementedError()
    
    @property
    def net_nodes(self):
        yield from self.hidden_nodes
        yield from self.output_nodes
        


