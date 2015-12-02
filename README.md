nnets
=====

Library for creating Neural Networks. A Neural Network can be created
with any architecture and activation functions. 

The network expression is computed using 



In particul....




```python
import sys
```


```python
sys.path.append("src/")
```

Create a network with one input node one sigmoid hidden layer
and a linear output layer


```python
import sympy
from nnets import mlp

sympy.init_printing()


simple_net = mlp.build_network(1,2,1)
simple_net.total_output_formula
```




$$- \theta_{20} + \frac{w^{20}_{10}}{e^{\theta_{10} - w^{10}_{00} x_{00}} + 1} + \frac{w^{20}_{11}}{e^{\theta_{11} - w^{11}_{00} x_{00}} + 1}$$



Iterate over the nodes


```python
for node in simple_net.all_nodes:
    print("Node of type {} with formula {}".format(type(node).__name__, node.formula))
```

    Node of type InputNode with formula x_00
    Node of type HiddenNode with formula 1/(exp(theta_10 - w_00__10*x_00) + 1)
    Node of type HiddenNode with formula 1/(exp(theta_11 - w_00__11*x_00) + 1)
    Node of type OutputNode with formula -theta_20 + w_10__20/(exp(theta_10 - w_00__10*x_00) + 1) + w_11__20/(exp(theta_11 - w_00__11*x_00) + 1)



```python
import numpy as np
x = np.arange(10000000)
#Initialize network with random parameters
params = np.random.rand(simple_net.nparams)
simple_net.parameters = params
simple_net(x)
```




    array([ 0.14626265,  0.30898658,  0.44234205, ...,  0.6559327 ,
            0.6559327 ,  0.6559327 ])



Compute the derivative of the NN expression with 
respect to the input symbol


```python
formula = simple_net.total_output_formula
formula.diff(next(simple_net.input_symbols))
```




$$\frac{w^{10}_{00} w^{20}_{10} e^{\theta_{10} - w^{10}_{00} x_{00}}}{\left(e^{\theta_{10} - w^{10}_{00} x_{00}} + 1\right)^{2}} + \frac{w^{11}_{00} w^{20}_{11} e^{\theta_{11} - w^{11}_{00} x_{00}}}{\left(e^{\theta_{11} - w^{11}_{00} x_{00}} + 1\right)^{2}}$$




```python
import sympy
```


```python
activation = sympy.sin
```


```python
output = mlp.OutputLayer(1, lambda x: x**2)
```


```python
n = mlp.build_network(1, layer, output)
```


```python

n.total_output_formula
```




$$\left(- \theta_{20} - w^{20}_{10} \sin{\left (\theta_{10} - w^{10}_{00} x_{00} \right )} - w^{20}_{11} \sin{\left (\theta_{11} - w^{11}_{00} x_{00} \right )} - w^{20}_{12} \sin{\left (\theta_{12} - w^{12}_{00} x_{00} \right )} - w^{20}_{13} \sin{\left (\theta_{13} - w^{13}_{00} x_{00} \right )} - w^{20}_{14} \sin{\left (\theta_{14} - w^{14}_{00} x_{00} \right )}\right)^{2}$$

