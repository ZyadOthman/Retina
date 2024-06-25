#ifndef TENSOR_C_INTERFACE_H
#define TENSOR_C_INTERFACE_H

#include <vector>
#include <string>
#include "Tensor.hpp"


#ifdef __cplusplus
extern "C" {
#endif

    // Forward declaration of Tensor structure
    typedef struct Tensor Tensor;

    // Function declarations
    Tensor* create_tensor(float* data, int* shape, int ndim, const char* device);
    float get_item(Tensor* tensor, float* indices);
    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* log_tensor(Tensor* tensor);
    Tensor* scalar_add_tensor(float scalar, Tensor* tensor);
    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* scalar_sub_tensor(float scalar, Tensor* tensor);
    Tensor* scalar_div_tensor(float scalar, Tensor* tensor);
    Tensor* scalar_mul_tensor(float scalar, Tensor* tensor);
    Tensor* tensor_div_scalar(Tensor* tensor, float scalar) ;
    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* tensor_pow_scalar(Tensor* tensor, float base);
    Tensor* scalar_pow_tensor(float exponent, Tensor* tensor);
    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* zeros_like_tensor(Tensor* tensor);
    Tensor* ones_like_tensor(Tensor* tensor);
    Tensor* transpose_tensor(Tensor* tensor);
    Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2);
    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    Tensor* to_device(Tensor* tensor, char* target_device);
    Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdim);
    void print_tensor(const Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_C_INTERFACE_H
