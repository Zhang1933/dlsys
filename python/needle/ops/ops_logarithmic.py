from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz=array_api.max(Z,self.axes,keepdims=True)
        sum_exp=array_api.sum(array_api.exp(Z-maxz),self.axes)
        return array_api.log(sum_exp)+maxz.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z=node.inputs[0].numpy()
        maxz=array_api.max(Z,self.axes,keepdims=True)
        sum_exp=array_api.sum(array_api.exp(Z-maxz),self.axes,keepdims=True)
        partial= array_api.exp(Z-maxz)/sum_exp
        out_grad_data=out_grad.numpy()
        if self.axes !=None:
            out_grad_data=array_api.expand_dims(out_grad_data,self.axes)
        partial_adjoint=partial*out_grad_data
        return (Tensor(partial_adjoint),)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

