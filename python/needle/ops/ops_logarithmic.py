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
        maxz=Z.max(self.axes,keepdims=True)
        sum_exp=array_api.sum(array_api.exp(Z-maxz.broadcast_to(Z.shape)),self.axes)
        return array_api.log(sum_exp)+maxz.reshape(sum_exp.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z=node.inputs[0].realize_cached_data()
        maxz=Z.max(self.axes,keepdims=True)
        sum_exp=array_api.sum(array_api.exp(Z-maxz.broadcast_to(Z.shape)),self.axes,keepdims=True).broadcast_to(Z.shape)
        partial= array_api.exp(Z-maxz.broadcast_to(Z.shape))/sum_exp
        out_grad_data=out_grad.realize_cached_data()
        if self.axes !=None:
            tmp_shape=list(out_grad_data.shape)
            for i in self.axes:
                tmp_shape.insert(i, 1)
            out_grad_data=out_grad_data.compact().reshape(tuple(tmp_shape)).broadcast_to(partial.shape)
        partial_adjoint=partial*out_grad_data
        return (Tensor(partial_adjoint,device=partial_adjoint.device),)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

# 