#include "tensor_c_interface.hpp"

extern "C" {

    Tensor* create_tensor(float* data, int* shape, int ndim, const char* device) {
        int size = 1;
        for (int i = 0; i < ndim; i++)
            size *= shape[i];

        std::unique_ptr<float[]> dataPtr(new float[size]);
        std::copy(data, data + size, dataPtr.get());

        std::vector<int> shape_vec(shape, shape + ndim);

        return new Tensor(std::move(dataPtr), shape_vec, device);
    }

    float get_item(Tensor* tensor, float* indices) {
        return tensor->get_item(std::vector<float>(indices, indices + tensor->get_ndim()));
    }

    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            std::cout << "Tensors must have the same number of dimensions " << tensor1->ndim << " and " << tensor2->ndim << " for addition" << std::endl;
            exit(1);
        }

        for (int i = 0; i < tensor1->ndim; i++) {
            if (tensor2->shape[i] !=  tensor2->shape[i]) {
                std::cout << "Tensors must have the same shape " << tensor1->shape[i] << " and " << tensor2->shape[i] << " at index " << i << " for addition" << std::endl;
                exit(1);
            }
        }
        if (tensor1->device == "cpu" && tensor2->device == "cpu")
            return tensor1->add_tensor_cpu(tensor2);
        else
            return tensor1->add_tensor_sycl(*tensor2);
    }


    Tensor* scalar_add_tensor(float scalar, Tensor* tensor) {
        if (tensor->device == "cpu")
            return tensor->scalar_add_tensor_cpu(scalar);
        else
            return tensor->scalar_add_tensor_sycl(scalar);
    }

    Tensor* scalar_sub_tensor(float scalar, Tensor* tensor) {
        if (tensor->device == "cpu")
            return tensor->scalar_sub_tensor_cpu(scalar);
        else
            return tensor->scalar_sub_tensor_sycl(scalar);
    }

    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            std::cout << "Tensors must have the same number of dimensions " << tensor1->ndim << " and " << tensor2->ndim << " for substraction" << std::endl;
            exit(1);
        }

        for (int i = 0; i < tensor1->ndim; i++) {
            if (tensor2->shape[i] !=  tensor2->shape[i]) {
                std::cout << "Tensors must have the same shape " << tensor1->shape[i] << " and " << tensor2->shape[i] << " at index " << i << " for substraction" << std::endl;
                exit(1);
            }
        }
        if (tensor1->device == "cpu" && tensor2->device == "cpu")
            return tensor1->sub_tensor_cpu(tensor2);
        else
            return tensor1->sub_tensor_sycl(*tensor2);
    }

    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->device == "cpu" && tensor2->device == "cpu")
            return tensor1->elementwise_mul_tensor_cpu(tensor2);
        else
            return tensor1->elementwise_mul_tensor_sycl(tensor2);
    }

    Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->device == "cpu" && tensor2->device == "cpu")
            return tensor1->elementwise_div_tensor_cpu(tensor2);
        else
            return tensor1->elementwise_div_tensor_sycl(tensor2);
    }

    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->device == "cpu" && tensor2->device == "cpu")
            return tensor1->matmul_tensor_cpu(tensor2);
        else {
             return tensor1->matmul_tensor_sycl(tensor2);
        }
    }

    Tensor* tensor_pow_scalar(Tensor* tensor, float base) {
        if (tensor->device == "cpu")
            return tensor->tensor_pow_scalar_cpu(base);
        else
            return tensor->tensor_pow_scalar_sycl(base);
        return tensor;
    }

    Tensor* scalar_pow_tensor(float exponent, Tensor* tensor) {
        if (tensor->device == "cpu")
            return tensor->scalar_pow_tensor_cpu(exponent);
        else
            return tensor->scalar_pow_tensor_sycl(exponent);
    }

    Tensor* scalar_div_tensor(float scalar, Tensor* tensor) {
        if (tensor->device == "cpu")
            return tensor->scalar_div_tensor_cpu(scalar);
        else
            return tensor->scalar_div_tensor_sycl(scalar);
    }

    Tensor* scalar_mul_tensor(float scalar, Tensor* tensor) {
        if (tensor->device == "cpu")
            return tensor->scalar_mul_tensor_cpu(scalar);
        else
            return tensor->scalar_mul_tensor_sycl(scalar);
    }

    Tensor* tensor_div_scalar(Tensor* tensor, float scalar) {
        if (tensor->device == "cpu")
            return tensor->tensor_div_scalar_cpu(scalar);
        else
            return tensor->tensor_div_scalar_sycl(scalar);
    }

    Tensor* log_tensor(Tensor* tensor) {
        if (tensor->device == "cpu")
            return tensor->log_tensor_cpu();
        else
            return tensor->log_tensor_sycl();
    }

    Tensor* zeros_like_tensor(Tensor* tensor){
        if (tensor->device == "cpu")
            tensor->zeros_like_tensor_cpu();
        else
            tensor->zeros_like_tensor_gpu();
        return tensor;
    }

    Tensor* ones_like_tensor(Tensor* tensor){
        if (tensor->device == "cpu")
            tensor->ones_like_tensor_cpu();
        else 
            tensor->ones_like_tensor_gpu();
        return tensor;
    }

    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
        std::vector<int> new_shape_vec(new_shape, new_shape + new_ndim);

        return tensor->reshape_tensor(new_shape_vec, new_ndim);
    }

    void print_tensor(const Tensor* tensor) {
        tensor->print();
    }

    Tensor* to_device(Tensor* tensor, char* target_device) {
        tensor->to_device(target_device);

        return tensor;
    }

    Tensor* transpose_tensor(Tensor* tensor) {
        if (tensor->device == "cpu") {
            switch (tensor->ndim) {
                case 1:
                    tensor->transpose_1D_tensor_cpu();
                    break;
                case 2:
                    tensor->transpose_2D_tensor_cpu();
                    break;
                case 3:
                    tensor->transpose_3D_tensor_cpu();
                    break;
                default:
                    std::cout << "Transpose only supports tensors up to 3 dimensions." << std::endl;
                    exit(1);
            }
        } else if (tensor->device == "sycl") {
            switch (tensor->ndim) {
                case 1:
                    tensor->transpose_1D_tensor_sycl();
                    break;
                case 2:
                    tensor->transpose_2D_tensor_sycl();
                    break;
                case 3:
                    tensor->transpose_3D_tensor_sycl();
                    break;
                default:
                    std::cout << "Transpose only supports tensors up to 3 dimensions." << std::endl;
                    exit(1);
            }
        } else {
            std::cerr << "Unsupported device type." << std::endl;
            exit(1);
        }
        return tensor;
    }

    Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2) {
        return tensor->transpose_axes_tensor(axis1, axis2);
    }

    Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdim) {
        if (tensor->size == 1)
            return tensor;
        if (tensor->device == "cpu")
            return tensor->sum_tensor_cpu(axis, keepdim);
        else
            return tensor->sum_tensor_sycl(axis, keepdim);
    }

}
