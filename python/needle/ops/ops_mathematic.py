"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return (out_grad*self.scalar*(a**(self.scalar-1)),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return (out_grad*self.scalar*(a**(self.scalar-1)),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):

        ### BEGIN YOUR SOLUTION
        # if not isinstance(node.inputs[0], NDArray) or not isinstance(
        #     node.inputs[1], NDArray
        # ):
        #     raise ValueError("Both inputs must be tensors (NDArray).")
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad/b
        grad_b = -1*out_grad*a*(b**-2)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a/self.scalar

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad/self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        lst=list(range(len(a.shape)))
        if self.axes==None:
            lst[-2],lst[-1]=lst[-1],lst[-2]
            return a.permute(tuple(lst))
        else:
            pos1=self.axes[0]
            pos2=self.axes[1]
            lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
            return a.permute(tuple(lst))

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.transpose(node.op.axes),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.reshape(node.inputs[0].shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a,self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes=[]
        input_shape=node.inputs[0].shape
        out_shape=self.shape
        out_shape_len=len(out_shape)
        i=0
        for i in range(1,len(input_shape)+1):
            if input_shape[-i]!=out_shape[-i]:
                axes.append(out_shape_len-i)
        if i !=out_shape_len:
            axes+=list(range(out_shape_len-i))

        return (reshape(summation(out_grad,tuple(axes)),input_shape),)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return a.sum(self.axes)
        elif isinstance( self.axes,tuple ):
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
        else:
            a = a.sum(axis =self.axes)
        return a
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        tmp_shape=list(out_grad.shape)
        if self.axes is not None:
            if isinstance(self.axes,int):
                tmp_shape.insert(self.axes,1)
            elif isinstance(self.axes,tuple):
                for axe in self.axes:
                    tmp_shape.insert(axe,1)
        else:
            tmp_shape=([1]*len(node.inputs[0].shape))
        tmp_shape=tuple(tmp_shape)
        return (out_grad.reshape(tmp_shape).broadcast_to(node.inputs[0].shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


def diff_axes(out_shape,input_shape):
    """
        输出两个shape长度不同的维度，input_shape长度<= out_shape
    """
    axes=[]
    i=0
    assert(len(input_shape)<=len(out_shape))
    for i in range(1,len(input_shape)+1):
        if input_shape[-i]!=out_shape[-i]:
            axes.append(i)
    if i !=len(out_shape):
        axes+=list(range(len(out_shape)-i))
    return tuple(axes)

class MatMul(TensorOp):
    def compute(self, a, b):
        """
            Z=X@W
        """
        return a@b

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x=node.inputs[0]
        w=node.inputs[1]
        x_adj=out_grad@(w.transpose())
        w_adj=x.transpose()@out_grad

        if w_adj.shape!=w.shape:
            w_adj=w_adj.sum(diff_axes(w_adj.shape,w.shape))
        if x_adj.shape !=x.shape:
            x_adj=x_adj.sum(diff_axes(x_adj.shape,x.shape))
        return x_adj,w_adj
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad/node.inputs[0],)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad*exp(node.inputs[0]),)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return (out_grad * Tensor(a > 0,device=out_grad.device),)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad_data=out_grad.realize_cached_data()
        input=node.inputs[0].realize_cached_data()
        return out_grad_data*((1-input.tanh()**2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        newshape=list(args[0].shape)
        l=len(args)
        newshape.insert(self.axis,l)
        res=array_api.empty(tuple(newshape),dtype=args[0].dtype,device=args[0].device)
        for i in range(l):
            idx=[slice(None,None,None)]*self.axis+[slice(i,i+1,1)]+[slice(None,None,None)]*(len(newshape)-self.axis-1)
            res[tuple(idx)]=args[i]
        return res
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        res=[]
        newshape=list(A.shape)
        newshape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            idx=[slice(None,None,None)]*self.axis+[slice(i,i+1,1)]+[slice(None,None,None)]*(len(A.shape)-self.axis-1)
            res.append(A[tuple(idx)].compact().reshape(tuple(newshape)))
        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad.cached_data=out_grad.realize_cached_data().flip(self.axes)
        return out_grad
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ## BEGIN YOUR SOLUTION
        newshape=list(a.shape)
        for axis in self.axes:
            newshape[axis]=a.shape[axis]*(self.dilation+1)
        res=array_api.full(tuple(newshape),0,device=a.device)
        idxes=[slice(None,None,None)]*len(a.shape)
        for axis in self.axes:
            idxes[axis]=slice(None,None,self.dilation+1)
        res[tuple(idxes)]=a
        return res
        ## END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape=newshape=list(a.shape)
        for axis in self.axes:
            newshape[axis]=a.shape[axis]//(self.dilation+1)
        res=array_api.full(tuple(newshape),0,device=a.device)
        idxes=[slice(None,None,None)]*len(a.shape)
        for axis in self.axes:
            idxes[axis]=slice(None,None,self.dilation+1)
        res=a[tuple(idxes)]
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A=A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C_in=A.shape
        K,_,_,C_out=B.shape
        Ns,Hs,Ws,Cs=A.strides
        A=A.as_strided(shape=(N,(H-K+1)//self.stride,(W-K+1)//self.stride,K,K,C_in),strides=(Ns,Hs*self.stride,Ws*self.stride,Hs,Ws,Cs)).compact().reshape((N*(H-K+1)//self.stride*(W-K+1)//self.stride,K*K*C_in))
        res=A@B.compact().reshape((K*K*C_in,C_out))
        return res.reshape((N,(H-K+1)//self.stride,(W-K+1)//self.stride,C_out)).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # V=conv(X,W)
        X,W=node.inputs
        K=W.shape[0]
        if self.stride>1:
            out_grad=dilate(out_grad,(1,2),self.stride-1)
        V_adj_permute=transpose(transpose(out_grad,(0,1)),(1,2))
        X_permute=transpose(X,(0,3))
        W_adj=conv(X_permute,V_adj_permute,padding=self.padding)
        W_adj=transpose(transpose(W_adj,(0,1)),(1,2))

        W_flip=flip(W,(0,1))
        W_trans=transpose(W_flip,(2,3))

        X_adj=conv(out_grad,W_trans,padding=K-1-self.padding)
        
        return X_adj,W_adj
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
