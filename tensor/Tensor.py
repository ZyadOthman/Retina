import ctypes
from autograd.functions import *

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Tensor:
    _C = ctypes.CDLL("tensor/libtensor.so")

    def __init__(self, data=None, device="cpu", requires_grad=False):
        
        self.device = device

        if data != None:
            data, shape = self.flatten(data)
            # Adjust the path to the shared library
            self.data_ctype = (ctypes.c_float * len(data))(*data)
            self.shape_ctype = (ctypes.c_int * len(shape))(*shape)
            self.ndim_ctype = ctypes.c_int(len(shape))
            self.device_ctype = device.encode('utf-8')

            self.shape = shape
            self.ndim = len(shape)
            self.numel = 1
            for s in self.shape:
                self.numel *= s
            self.requires_grad = requires_grad
            self.grad = None
            self.grad_fn = None

            Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
            Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)
            
            self.tensor = Tensor._C.create_tensor(
                self.data_ctype,
                self.shape_ctype,
                self.ndim_ctype,
                self.device_ctype
            )
        
        else:
            self.tensor = None,
            self.shape = None,
            self.ndim = None,
            self.numel = None,
            self.requires_grad = None
            self.grad = None
            self.grad_fn = None
        
    def flatten(self, nested_list):
        """
        This method simply convert a list type tensor to a flatten tensor with its shape
        
        Example:
        
        Arguments:  
            nested_list: [[1, 2, 3], [-5, 2, 0]]
        Return:
            flat_data: [1, 2, 3, -5, 2, 0]
            shape: [2, 3]
        """
        def flatten_recursively(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = flatten_recursively(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape
        
        flat_data, shape = flatten_recursively(nested_list)
        return flat_data, shape
    
    def ones_like(self):
        Tensor._C.ones_like_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.ones_like_tensor.restype = ctypes.POINTER(CTensor)
        Tensor._C.ones_like_tensor(self.tensor)   

        result_tensor_ptr = Tensor._C.ones_like_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel
        
        return result_data
    
    def zeros_like(self):
        Tensor._C.zeros_like_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.zeros_like_tensor.restype = ctypes.POINTER(CTensor)
        Tensor._C.zeros_like_tensor(self.tensor)  

        result_tensor_ptr = Tensor._C.zeros_like_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        return result_data
    
    def unsqueeze(self, dim):
        if dim < 0:
            dim = self.ndim + dim + 1

        # Ensure the dimension is valid
        if dim > self.ndim:
            raise ValueError("Dimension out of range (expected to be in range of [0, {0}], but got {1})".format(self.ndim, dim))
        
        # Create the new shape with an extra dimension of size 1
        new_shape = self.shape[:dim] + [1] + self.shape[dim:]
        
        return self.reshape(new_shape)
    
    def squeeze(self, dim=None):
        if dim is not None:
            if dim < 0:
                dim = self.ndim + dim
            
            # Ensure the dimension is valid
            if dim >= self.ndim or dim < 0:
                raise ValueError("Dimension out of range (expected to be in range of [0, {0}), but got {1})".format(self.ndim, dim))
            
            # Only squeeze the specified dimension if its size is 1
            if self.shape[dim] != 1:
                return self
                #raise ValueError("Dimension {0} does not have size 1 and cannot be squeezed".format(dim))
            
            # Create the new shape without the specified dimension
            new_shape = self.shape[:dim] + self.shape[dim+1:]
        else:
            # Create the new shape by removing all dimensions of size 1
            new_shape = [s for s in self.shape if s != 1]
        
        return self.reshape(new_shape)
    
    def __str__(self):
        def print_recursively(tensor, depth, index):
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor.shape[-1]):
                    index[-1] = i
                    result += str(tensor[tuple(index)]) + ", "
                return result.strip()
            else:
                result = ""
                if depth > 0:
                    result += "\n" + " " * ((depth - 1) * 4)
                for i in range(tensor.shape[depth]):
                    index[depth] = i
                    result += "["
                    result += print_recursively(tensor, depth + 1, index) + "],"
                    if i < tensor.shape[depth] - 1:
                        result += "\n" + " " * (depth * 4)
                return result.strip(",")

        index = [0] * self.ndim
        result = "tensor(["
        result += print_recursively(self, 0, index)
        result += f"""], device="{self.device}", requires_grad={self.requires_grad})"""
        return result

    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, indices):
        """
        Access tensor by index tensor[i, j, k...]
        """
        if isinstance(indices, int):
            # If indices is a single integer, convert it to a tuple of one element
            indices = (indices,)
        elif isinstance(indices, list):
            # If indices is a list, convert it to a tuple
            indices = tuple(indices)
        elif isinstance(indices, tuple):
            if len(indices) != self.ndim:
                raise ValueError("Number of indices must match the number of dimensions")
        else:
            raise TypeError("Indices must be an int, a list of ints, or a tuple of ints")

        # Define argument and return types for the C function
        Tensor._C.get_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float

        # Convert indices to ctypes array
        indices_array = (ctypes.c_int * len(indices))(*indices)

        # Call the C function to get the item
        value = Tensor._C.get_item(self.tensor, indices_array)

        return value

    def reshape(self, new_shape):
        """
        Reshape tensor
        result = tensor.reshape([1,2])
        """
        new_shape_ctype = (ctypes.c_int * len(new_shape))(*new_shape)
        new_ndim_ctype = ctypes.c_int(len(new_shape))
        
        Tensor._C.reshape_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C.reshape_tensor.restype = ctypes.POINTER(CTensor)
        result_tensor_ptr = Tensor._C.reshape_tensor(self.tensor, new_shape_ctype, new_ndim_ctype)   

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = new_shape.copy()
        result_data.ndim = len(new_shape)
        result_data.device = self.device
        result_data.numel = self.numel

        return result_data
    
    def __radd__(self, other):
        other = float(other)
        Tensor._C.scalar_add_tensor.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor)]
        Tensor._C.scalar_add_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.scalar_add_tensor(ctypes.c_float(other), self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel


        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = AddBackward(other, self)

        return result_data

    def __add__(self, other):
        """
        Add tensors
        result = tensor1 + tensor2
        """
    
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for addition")
        
        Tensor._C.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad or other.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = AddBackward(self, other)

        return result_data
    
    def __sub__(self, other):
        """
        Sub tensors
        result = tensor1 - tensor2
        """
        if isinstance(other, (int, float)):
            other = float(other)
            Tensor._C.scalar_sub_tensor.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor)]
            Tensor._C.scalar_sub_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.scalar_sub_tensor(ctypes.c_float(other), self.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim
            result_data.device = self.device
            result_data.numel = self.numel

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = SubBackward(self, other)

            return result_data 
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape for substraction")
            
            Tensor._C.sub_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.sub_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.sub_tensor(self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim
            result_data.device = self.device
            result_data.numel = self.numel

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = SubBackward(self, other)

            return result_data


    def __rsub__(self, other):
        other = float(other)
        Tensor._C.scalar_sub_tensor.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor)]
        Tensor._C.scalar_sub_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.scalar_sub_tensor(ctypes.c_float(other), self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad or other.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = SubBackward(self, other)

        return result_data 


    def print_tensor(self):
        Tensor._C.print_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.print_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.print_tensor(self.tensor)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result_data = Tensor()
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim
            result_data.device = self.device
            result_data.numel = self.numel

            Tensor._C.scalar_mul_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
            Tensor._C.scalar_mul_tensor.restype = ctypes.POINTER(CTensor)

            result_data.tensor = Tensor._C.scalar_mul_tensor(self.tensor, ctypes.c_float(other))

            result_data.requires_grad = self.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = ScalarMulBackward(self, other)

            return result_data
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape for element-wise multiplication")

            Tensor._C.elementwise_mul_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.elementwise_mul_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.elementwise_mul_tensor(self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim
            result_data.device = self.device
            result_data.numel = self.numel

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = ElementwiseMulBackward(self, other)

            return result_data
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)  

    def __pos__(self):
        return self  
    
    def __matmul__(self, other):
        #2D matmul
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")

        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible shapes for matrix multiplication")

        Tensor._C.matmul_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
        Tensor._C.matmul_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.matmul_tensor(self.tensor, other.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = [self.shape[0], other.shape[1]]
        result_data.ndim = 2
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad or other.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = MatmulBackward(self, other)
        return result_data
    
    def __pow__(self, other):
        other = float(other)
        Tensor._C.tensor_pow_scalar.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
        Tensor._C.tensor_pow_scalar.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.tensor_pow_scalar(self.tensor, ctypes.c_float(other))

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = PowBackward(self, other)
        
        return result_data
    
    def __rpow__(self, other):
        other = float(other)
        Tensor._C.scalar_pow_tensor.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor)]
        Tensor._C.scalar_pow_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.scalar_pow_tensor(ctypes.c_float(other), self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = PowBackward(other, self)

        return result_data
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = float(other)
            Tensor._C.tensor_div_scalar.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
            Tensor._C.tensor_div_scalar.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.tensor_div_scalar(self.tensor, ctypes.c_float(other))

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim
            result_data.device = self.device
            result_data.numel = self.numel
            
            result_data.requires_grad = self.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = DivisionBackward(self, other)
        
        elif isinstance(self, Tensor) and isinstance(other, Tensor):
            Tensor._C.tensor_div_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.tensor_div_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.tensor_div_tensor(self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim
            result_data.device = self.device
            result_data.numel = self.numel

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = DivisionBackward(self, other)

        return result_data
        
        
    
    def __rtruediv__(self, other):
        other = float(other)

        Tensor._C.scalar_div_tensor.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor)]
        Tensor._C.scalar_div_tensor.restype = ctypes.POINTER(CTensor)
        result_tensor_ptr = Tensor._C.scalar_div_tensor(ctypes.c_float(other), self.tensor)        

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = DivisionBackward(other, self)

        return result_data
    
    def to(self, device):
        self.device = device
        self.device_ctype = self.device.encode('utf-8')
    
        Tensor._C.to_device.argtypes = [ctypes.POINTER(CTensor), ctypes.c_char_p]
        Tensor._C.to_device.restype = None
        Tensor._C.to_device(self.tensor, self.device_ctype)
    
        return self
    
    def log(self):
        Tensor._C.log_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.log_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.log_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = LogBackward(self)

        return result_data
    
    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.shape == [1]:
                gradient = Tensor([1])
            else:
                raise RuntimeError("Gradient argument must be specified for non-scalar tensors.")
            
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad += gradient


        if self.grad_fn is not None:
            grads = self.grad_fn.backward(gradient)
            for tensor, grad in zip(self.grad_fn.input, grads):
                if isinstance(tensor, Tensor):
                    tensor.backward(grad) # recursively call the backward again for the gradient expression (chain rule)
        
    def zero_grad(self):
        self.grad = None

    def detach(self):
        self.grad = None
        self.grad_fn = None

    def transpose(self, axis1, axis2):
        if axis1 < 0:
            axis1 = self.ndim + axis1
        if axis2 < 0:
            axis2 = self.ndim + axis2

        Tensor._C.transpose_axes_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]
        Tensor._C.transpose_axes_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.transpose_axes_tensor(self.tensor, axis1, axis2)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape[::-1].copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = TransposeBackward(self, axis1, axis2)

        return result_data   

    @property
    def T(self):
        Tensor._C.transpose_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.transpose_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.transpose_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()[::-1]
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = TBackward(self)

        return result_data
    
    def sum(self, axis=None, keepdim=False):
            if axis is not None and axis < 0:
                axis = self.ndim + axis
                
            if axis == None:
                axis = -1

            if axis > self.ndim - 1:
                raise ValueError(f"Error: axis argument {axis} cannot be higher than tensor dimension {self.ndim}")

            Tensor._C.sum_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_bool]
            Tensor._C.sum_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.sum_tensor(self.tensor, axis, keepdim)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr

            if axis == -1:
                if keepdim:
                    result_data.ndim = self.ndim
                    result_data.shape = [1] * self.ndim

                else:
                    result_data.shape = [1]
                    result_data.ndim = 1
            else:
                if keepdim:
                    result_data.shape = self.shape[:axis] + [1] + self.shape[axis+1:]
                else:
                    result_data.shape = self.shape[:axis] + self.shape[axis+1:]
                result_data.ndim = len(result_data.shape)

            result_data.device = self.device
            result_data.numel = 1
            for s in result_data.shape:
                result_data.numel *= s

            result_data.requires_grad = self.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = SumBackward(self, axis, keepdim=keepdim)

            return result_data