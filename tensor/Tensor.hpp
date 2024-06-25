#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <CL/sycl.hpp>

class vector_addition;
class vector_subtraction;

class Tensor {
public:
    Tensor(std::unique_ptr<float[]>&& data, std::vector<int>& shape, const std::string& device);
    ~Tensor();

    const float get_item(std::vector<float> indices);
    const int get_ndim();

    Tensor *add_tensor_cpu(Tensor* tensor);
    Tensor *scalar_add_tensor_cpu(float scalar);
    Tensor *sub_tensor_cpu(Tensor* tensor);
    Tensor *scalar_sub_tensor_cpu(float scalar);
    Tensor *scalar_div_tensor_cpu(float scalar);
    Tensor *tensor_div_scalar_cpu(float scalar);
    Tensor *elementwise_mul_tensor_cpu(Tensor* tensor);
    Tensor *elementwise_div_tensor_cpu(Tensor* tensor);
    Tensor *matmul_tensor_cpu(const Tensor* tensor);
    Tensor *scalar_pow_tensor_cpu(float base);
    Tensor *scalar_mul_tensor_cpu(float scalar);
    Tensor *log_tensor_cpu();
    Tensor *tensor_pow_scalar_cpu(float exponent);

    Tensor *add_tensor_sycl(Tensor& tensor2);
    Tensor *log_tensor_sycl();
    Tensor *scalar_add_tensor_sycl(float scalar);
    Tensor *scalar_sub_tensor_sycl(float scalar);
    Tensor *sub_tensor_sycl(Tensor& tensor2);
    Tensor *scalar_pow_tensor_sycl(float base);
    Tensor *scalar_div_tensor_sycl(float scalar);
    Tensor *tensor_div_scalar_sycl(float scalar);
    Tensor *elementwise_mul_tensor_sycl(Tensor* tensor);
    Tensor *elementwise_div_tensor_sycl(Tensor* tensor);
    Tensor *tensor_pow_scalar_sycl(float exponent);
    Tensor *scalar_mul_tensor_sycl(float scalar);
    Tensor *matmul_tensor_sycl(Tensor* tensor2);

    void zeros_like_tensor_cpu();
    void ones_like_tensor_cpu();

    void zeros_like_tensor_gpu();
    void ones_like_tensor_gpu();

    void transpose_1D_tensor_cpu(); 
    void transpose_2D_tensor_cpu(); 
    void transpose_3D_tensor_cpu(); 

    void transpose_1D_tensor_sycl(); 
    void transpose_2D_tensor_sycl(); 
    void transpose_3D_tensor_sycl(); 

    Tensor *transpose_axes_tensor(int axis1, int axis2);

    Tensor *sum_tensor_cpu(int axis, bool keepdim);
    Tensor *sum_tensor_sycl(int axis, bool keepdim);

    Tensor *reshape_tensor(std::vector<int>& new_shape, int new_ndim);
    void to_device(const std::string& target_device);
    void print() const;

    std::unique_ptr<float[]> data;
    std::optional<cl::sycl::buffer<float, 1>> data_gpu;
    std::vector<int> shape;
    std::vector<int> strides;
    int ndim;
    int size;
    std::string device;
};



void cpu_to_sycl(Tensor& tensor, cl::sycl::queue& q);
void sycl_to_cpu(Tensor& tensor, cl::sycl::queue& q);

#endif // TENSOR_HPP
