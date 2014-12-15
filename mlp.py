# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:19:45 2014

@author: zah
"""
import abc
from collections import Sequence
from itertools import chain, repeat


import numpy as np

from nnets.neural_networks import (NeuralNetwork, HiddenNode, InputNode, OutputNode,
                             sigmoid_g, linear_g                            
                             )


class NetworkPartError(Exception):
    pass

class NetworkPart():
    _parts = []

    def __init__(self, parts  = None):
        if parts is None:
            parts = []
        self._parts = list(parts)
        self.check_consistent()

    @property
    def _starts_with_connector(self):
        return self._parts and isinstance(self._parts[0], Connector)

    @property
    def _ends_with_connector(self):
        return self._parts and isinstance(self._parts[-1], Connector)


    @property
    def connectors(self):
        if self._starts_with_connector:
            return self._parts[0::2]
        else:
            return self._parts[1::2]

    @property
    def layers(self):
        if self._starts_with_connector:
            return self._parts[1::2]
        else:
            return self._parts[0::2]

    @property
    def is_complete(self):
        layers = self.layers
        return (layers and isinstance(layers[0], InputLayer)
                and isinstance(layers[-1], OutputLayer))


    def check_consistent(self):
        layers = self.layers
        connectors = self.connectors
        laylen = len(layers)
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError("%s is not an instance of Layer." % layer)
            if isinstance(layer, InputLayer):
                if i>0:
                    raise TypeError("Input Layers must be at the beginning.")
                elif self._starts_with_connector:
                    raise TypeError("Input Layers must not "
                                    "have a connector prepended.")

            elif isinstance(layer, OutputLayer):
                if i<(laylen-1):
                    raise TypeError("Output Layers must be at the end.")
                elif self._ends_with_connector:
                    raise TypeError("Output Layers must not "
                                    "have a connector appended.")


        for i, connector in enumerate(connectors):
            if not isinstance(connector, Connector):
                raise TypeError("%s is not an instance of Connector"
                                % connector)


        if [o for o in layers[:-1] if isinstance(o, OutputLayer)]:
            raise TypeError("Output Layers must be at the end.")

    def append_part(self, part, connector = None):
        if not part._parts:
            parts = self._parts
        elif not self._parts:
            parts = part._parts
        elif not self._ends_with_connector and not part._starts_with_connector:
            if connector is None:
                connector = CompleteConnector()
            else:
                try:
                    if issubclass(connector, Connector):
                         connector = connector()
                except TypeError:
                    pass
            parts = self._parts + [connector] + part._parts
        else:
            parts = self._parts + part._parts

        return NetworkPart(parts)



    def __add__(self, other):
        newpart = self.append_part(other)
        try:
            return MultiLayerNetwork(newpart)
        except NetworkPartError:
            return newpart


class NetworkPiece(NetworkPart):
    def __init__(self):
        self._parts = [self]


class Connector(NetworkPiece, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def connect(self, layer1, layer2):
        """Attach layer 2 to layer 1"""
        return NotImplemented


class CompleteConnector(Connector):
    def connect(self, layer1, layer2):
        for node in layer2:
            node.inputs = [node1 for node1 in layer1]

class RandomConnector(Connector):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def connect(self, layer1, layer2):
        for node in layer2:
            node.inputs = [node1 for node1 in layer1
                            if np.random.rand() > self.p]
class Layer(NetworkPiece):

    node_type = HiddenNode

    def __init__(self, number=1, function_spec = sigmoid_g):
        super().__init__()
        self.number = number
        if isinstance(function_spec, Sequence):
            if len(function_spec) != number:
                raise ValueError("Function spec should be one "
                                "callable or a sequence of length `number`")
        else:
            self.function_spec = [function_spec] * number

    def init_nodes(self, names):
        self.nodes = [self.node_type(name, function) for
                        name, function in zip(names, self.function_spec)]

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

class InputLayer(Layer):
    node_type = InputNode
    def __init__(self, number=1, function_spec = linear_g):
        super().__init__( number, function_spec)

class OutputLayer(Layer):
    node_type = OutputNode
    def __init__(self, number=1, function_spec = linear_g):
        super().__init__( number, function_spec)
    
class MultiLayerNetwork(NeuralNetwork):
    def __init__(self, network_part):
        if not network_part.is_complete:
            raise NetworkPartError("Network Part Specification "
                                    "is not complete.")
        self.make_net(network_part)
        super().__init__()
        
    def make_net(self, network_part):
        layers =  network_part.layers
        connectors = network_part.connectors
        layerpairs = zip(layers, layers[1:])
        for i, layer in enumerate(layers):
            #TODO: Add separator.
            names = ('%i%i' % (i, j) for j in range(layer.number))
            layer.init_nodes(names)
        for connector, layerpair in zip(connectors, layerpairs):
            connector.connect(*layerpair)

        self.layers = layers

    @property
    def input_nodes(self):
        yield from self.layers[0]


    @property
    def output_nodes(self):
        yield from self.layers[-1]

    @property
    def all_nodes(self):
        yield from chain(*self.layers)

    @property
    def hidden_nodes(self):
        yield from chain(*self.layers[1:-1])

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__,
                    ','.join(str(len(layer)) for layer in self.layers))

def build_network(*args):
    l = len(args)
    if l<2:
        raise TypeError("At least 2 arguments are expected.")
    parts = []
    types = chain((InputLayer,), repeat(Layer, l-2), (OutputLayer,))
    for arg,cls in zip(args, types):
        if isinstance(arg, int):
            part = cls(arg)
        elif isinstance(arg, (Connector, cls)):
            part = arg
        else:
            raise TypeError("Element %r not understood" % arg)
        parts += [part]
    return sum(parts, NetworkPart())