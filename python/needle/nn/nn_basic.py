"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(in_features,out_features,dtype=dtype))
        if bias:
            self.bias=Parameter(init.kaiming_uniform(out_features,1,dtype=dtype).reshape((1,out_features)))
        else:
            self.bias=None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y=X@self.weight
        if self.bias:
            y+=self.bias.broadcast_to(shape=y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        non_batch_dim_len=1
        for i in X.shape[1:]:
            non_batch_dim_len*=i
        return X.reshape((X.shape[0],non_batch_dim_len))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x=module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot=init.one_hot(logits.shape[-1],y)
        label_logit=(one_hot*logits).sum((len(logits.shape)-1,))
        return  (ops.logsumexp(logits,axes=(1,))-label_logit).sum(0)/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim,device=device,dtype=dtype))
        self.bias=Parameter(init.zeros(dim,device=device,dtype=dtype))
        self.running_mean=init.zeros(dim,device=device,dtype=dtype)
        self.running_var=init.ones(dim,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            Ex=x.sum((0,))/x.shape[0]
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*Ex.data
            Ex=Ex.reshape((1,x.shape[1])).broadcast_to(x.shape)

            var=((x-Ex)*(x-Ex)).sum((0,))/x.shape[0]
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*var.data
            var=var.reshape((1,x.shape[1])).broadcast_to(x.shape)
            
            y=self.weight.broadcast_to(x.shape)*((x-Ex)/(var+self.eps)**(0.5))+self.bias.broadcast_to(x.shape)
        else:
            y=self.weight.broadcast_to(x.shape)*(x-self.running_mean.reshape((1,x.shape[1])))/((self.running_var.reshape((1,x.shape[1]))+self.eps)**0.5)+self.bias.broadcast_to(x.shape)
        return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim,device=device,dtype=dtype))
        self.bias=Parameter(init.zeros(dim,device=device,dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # needle没有默认广播机制，反向传播的时候有问题
        Ex=(x.sum((1,))/x.shape[1]).reshape((x.shape[0],1)).broadcast_to(x.shape)
        Var=(((x-Ex)*(x-Ex)).sum((1,))/x.shape[1]).reshape((x.shape[0],1)).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape)*(x-Ex)/((Var+self.eps)**(0.5))+self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            x=x*init.randb(*x.shape,p=1-self.p,device=x.device)/(1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn.forward(x)+x
        ### END YOUR SOLUTION
