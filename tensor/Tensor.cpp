#include "Tensor.hpp"

Tensor::Tensor(std::unique_ptr<float[]>&& data, std::vector<int>& shape, const std::string& device)
    : data(std::move(data)), shape(shape), ndim(shape.size()), device(device) {
    
    size = 1;
    for (int dim : shape) {
        size *= dim;
    }

    strides.resize(ndim);
    int stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    if (device == "sycl") {
        data_gpu = cl::sycl::buffer<float, 1>(this->data.get(), cl::sycl::range<1>(size));
    }
}

Tensor::~Tensor() {
    // Destructor implementation (if needed)
}

const float Tensor::get_item(std::vector<float> indices) {
    int index = 0;
    for (int i = 0; i < ndim; i++)
        index += indices[i] * strides[i];
    
    float result;
    result = data[index];

    return result;
}

const int Tensor::get_ndim(){
    return ndim;
}

Tensor *Tensor::add_tensor_cpu(Tensor *tensor) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] = data[i] + tensor->data[i];
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::scalar_add_tensor_cpu(float scalar) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] = data[i] + scalar;
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::scalar_div_tensor_cpu(float scalar) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] =  scalar / data[i];
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::scalar_mul_tensor_cpu(float scalar) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] =  scalar * data[i];
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::tensor_div_scalar_cpu(float scalar) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] = data[i] / scalar;
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::scalar_sub_tensor_cpu(float scalar) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] =  data[i] - scalar;
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::sub_tensor_cpu(Tensor* tensor) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] = data[i] - tensor->data[i];
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::elementwise_mul_tensor_cpu(Tensor* tensor) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] = data[i] * tensor->data[i];
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::elementwise_div_tensor_cpu(Tensor* tensor) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        result_data[i] = data[i] / tensor->data[i];
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::scalar_pow_tensor_cpu(float base) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++) {
        result_data[i] = powf(base, data[i]);
    }
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::tensor_pow_scalar_cpu(float exponent) {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++) {
        result_data[i] = powf(data[i], exponent);
    }
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::matmul_tensor_cpu(const Tensor* tensor) {
    if (ndim != 2 || tensor->ndim != 2) {
        std::cerr << "Matrix multiplication requires both tensors to be 2-dimensional." << std::endl;
        return nullptr;
    }
    if (shape[1] != tensor->shape[0]) {
        std::cerr << "Cannot multiply tensors. Inner dimensions do not match." << std::endl;
        return nullptr;
    }

    int m = shape[0];
    int n = shape[1];
    int k = tensor->shape[1];
    std::unique_ptr<float[]> result_data = std::make_unique<float[]>(m * k);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            result_data[i * k + j] = 0.0f;
            for (int l = 0; l < n; ++l) {
                result_data[i * k + j] += data[i * n + l] * tensor->data[l * k + j];
            }
        }
    }

    std::vector<int> new_shape = {m, k};
    return new Tensor(std::move(result_data), new_shape, device);
}

Tensor *Tensor::sum_tensor_cpu(int axis, bool keepdim) {
    if (axis < 0) {
        axis += ndim;
    }

    if (axis >= ndim || axis < 0) {
        std::cerr << "Invalid axis" << std::endl;
        return nullptr;
    }

    std::vector<int> new_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) {
            new_shape.push_back(shape[i]);
        } else if (keepdim) {
            new_shape.push_back(1);
        }
    }

    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }

    std::unique_ptr<float[]> result_data(new float[new_size]());
    
    for (int i = 0; i < size; ++i) {
        int index = i;
        int remainder = i;
        int new_index = 0;
        int new_stride = 1;
        for (int j = ndim - 1; j >= 0; --j) {
            int current_dim = shape[j];
            int position = remainder % current_dim;
            remainder /= current_dim;
            if (j != axis) {
                new_index += position * new_stride;
                new_stride *= shape[j];
            }
        }
        result_data[new_index] += data[i];
    }

    return new Tensor(std::move(result_data), new_shape, device);
}

Tensor *Tensor::log_tensor_cpu() {
    auto result_data = std::make_unique<float[]>(size);
    for (int i = 0; i < size; ++i) {
        result_data[i] = std::log(data[i]);
    }
    std::cout << "Successfully computed log on tensor using CPU." << std::endl;
    return new Tensor(std::move(result_data), shape, device);
}

Tensor *Tensor::log_tensor_sycl() {
    if (!data_gpu) {
        std::cerr << "data_gpu is not initialized." << std::endl;
        return nullptr;
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    cl::sycl::buffer<float, 1> result_buffer(result_data.get(), cl::sycl::range<1>(size));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class log_tensor>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            result_acc[idx] = sycl::log(data_acc[idx]);
        });
    }).wait();


    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}



Tensor *Tensor::sum_tensor_sycl(int axis, bool keepdim) {
    if (axis < 0) {
        axis += ndim;
    }

    if (axis >= ndim || axis < 0) {
        std::cerr << "Invalid axis" << std::endl;
        return nullptr;
    }

    if (!data_gpu) {
        std::cerr << "data_gpu is not initialized." << std::endl;
        return nullptr;
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    std::vector<int> new_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) {
            new_shape.push_back(shape[i]);
        } else if (keepdim) {
            new_shape.push_back(1);
        }
    }

    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }

    auto result_data = std::make_unique<float[]>(new_size);
    cl::sycl::buffer<float, 1> result_buffer(result_data.get(), cl::sycl::range<1>(new_size));

    std::vector<int> shape_copy = shape;
    std::vector<int> strides_copy = strides;
    int* shape_raw = shape_copy.data();
    int* strides_raw = strides_copy.data();

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        int ndim_local = ndim;
        int axis_local = axis;

        cgh.parallel_for<class sum_tensor>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            int index = idx[0];
            int remainder = index;
            int new_index = 0;
            int new_stride = 1;

            for (int j = ndim_local - 1; j >= 0; --j) {
                int current_dim = shape_raw[j];
                int position = remainder % current_dim;
                remainder /= current_dim;
                if (j != axis_local) {
                    new_index += position * new_stride;
                    new_stride *= shape_raw[j];
                }
            }

            cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device> atomic_data(result_acc[new_index]);
            atomic_data.fetch_add(data_acc[index]);
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), new_shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::add_tensor_sycl(Tensor& tensor2) {
    if (size != tensor2.size) {
        std::cerr << "Cannot add tensors. Sizes are not compatible." << std::endl;
        return nullptr;
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    cl::sycl::buffer<float, 1> result_buffer(result_data.get(), cl::sycl::range<1>(size));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data1_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto data2_acc = tensor2.data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class vector_addition>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            result_acc[idx] = static_cast<float>(data1_acc[idx]) + static_cast<float>(data2_acc[idx]);
        });
    }).wait();


    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::scalar_add_tensor_sycl(float scalar) {
    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    cl::sycl::buffer<float, 1> result_buffer(result_data.get(), cl::sycl::range<1>(size));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class scalar_addition>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            result_acc[idx] = data_acc[idx] + scalar;
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::sub_tensor_sycl(Tensor& tensor2) {
    if (size != tensor2.size) {
        std::cerr << "Cannot subtract tensors. Sizes are not compatible." << std::endl;
        return nullptr;
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    cl::sycl::buffer<float, 1> result_buffer(result_data.get(), cl::sycl::range<1>(size));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data1_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto data2_acc = tensor2.data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class vector_subtraction>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            result_acc[idx] = static_cast<float>(data1_acc[idx]) - static_cast<float>(data2_acc[idx]);
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::scalar_pow_tensor_sycl(float base) {
    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    cl::sycl::buffer<float, 1> result_buffer(result_data.get(), cl::sycl::range<1>(size));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class scalar_pow_tensor>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            result_acc[idx] = std::pow(base, static_cast<float>(data_acc[idx]));
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::elementwise_mul_tensor_sycl(Tensor* tensor) {
    if (size != tensor->size) {
        std::cerr << "Tensors must be of the same size for element-wise multiplication." << std::endl;
        return nullptr;
    }

    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));


    if (!data_gpu || !tensor->data_gpu) {
        std::cerr << "SYCL buffers must be initialized before using them." << std::endl;
        return nullptr;
    }

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto tensor_data_acc = tensor->data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class elementwise_mul>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = data_acc[idx] * tensor_data_acc[idx];
        });
    }).wait();


    Tensor* result = new Tensor(std::move(data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::elementwise_div_tensor_sycl(Tensor* tensor) {
    if (size != tensor->size) {
        std::cerr << "Tensors must be of the same size for element-wise division." << std::endl;
        return nullptr;
    }

    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto tensor_data_acc = tensor->data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class elementwise_div>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = data_acc[idx] / tensor_data_acc[idx];
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::scalar_div_tensor_sycl(float scalar) {
    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class scalar_division>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = scalar / data_acc[idx];
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::scalar_sub_tensor_sycl(float scalar) {
    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class scalar_sub>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = scalar - data_acc[idx];
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::scalar_mul_tensor_sycl(float scalar) {
    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class scalar_multiplication>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = scalar * data_acc[idx];
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::tensor_div_scalar_sycl(float scalar) {
    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class tensor_division>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = data_acc[idx] / scalar;
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::tensor_pow_scalar_sycl(float exponent) {
    sycl::queue queue(sycl::gpu_selector_v);

    auto result_data = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> result_buffer(result_data.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_gpu->get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class tensor_pow_scalar>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            result_acc[idx] = std::pow(data_acc[idx], exponent);
        });
    }).wait();

    Tensor* result = new Tensor(std::move(result_data), shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}

Tensor *Tensor::matmul_tensor_sycl(Tensor* tensor2) {
    if (ndim != 2 || tensor2->ndim != 2) {
        std::cerr << "Matrix multiplication requires both tensors to be 2-dimensional." << std::endl;
        return nullptr;
    }
    
    if (shape[1] != tensor2->shape[0]) {
        std::cerr << "Cannot multiply tensors. Inner dimensions do not match: " << shape[0] << "," << shape[1] << " " << tensor2->shape[0] << "," << tensor2->shape[1] << std::endl;
        return nullptr;
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);

    int m = shape[0];
    int n = shape[1];
    int k = tensor2->shape[1];
    std::unique_ptr<float[]> result_data = std::make_unique<float[]>(m * k);
    auto result_buffer = cl::sycl::buffer<float, 1>(result_data.get(), cl::sycl::range<1>(m * k));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto data1_acc = data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto data2_acc = tensor2->data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        auto result_acc = result_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        cgh.parallel_for<class matrix_multiplication>(cl::sycl::range<2>(m, k), [=](cl::sycl::id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) {
                sum += data1_acc[row * n + i] * data2_acc[i * k + col];
            }
            result_acc[row * k + col] = sum;
        });
    }).wait();

    std::vector<int> new_shape = {m, k};

    Tensor* result = new Tensor(std::move(result_data), new_shape, device);
    result->data_gpu = std::make_optional<cl::sycl::buffer<float, 1>>(result_buffer);

    return result;
}


void Tensor::zeros_like_tensor_cpu(){
    auto data_host = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        data_host[i] = 0.0;
    data = std::move(data_host);
}

void Tensor::ones_like_tensor_cpu(){
    auto data_host = std::make_unique<float[]>(size);
    for (int i = 0; i < size; i++)
        data_host[i] = 1.0;
    data = std::move(data_host);
}

void Tensor::zeros_like_tensor_gpu() {
    if (!data_gpu) {
        data_gpu = cl::sycl::buffer<float, 1>(cl::sycl::range<1>(size));
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);
    queue.submit([&](cl::sycl::handler& h) {
        auto acc = data_gpu->get_access<cl::sycl::access::mode::write>(h);
        h.parallel_for(cl::sycl::range<1>(size), [=](cl::sycl::id<1> i) {
            acc[i] = 0.0f;
        });
    }).wait();

}

void Tensor::ones_like_tensor_gpu() {
    if (!data_gpu) {
        data_gpu = cl::sycl::buffer<float, 1>(cl::sycl::range<1>(size));
    }

    cl::sycl::queue queue(cl::sycl::gpu_selector_v);
    queue.submit([&](cl::sycl::handler& h) {
        auto acc = data_gpu->get_access<cl::sycl::access::mode::write>(h);
        h.parallel_for(cl::sycl::range<1>(size), [=](cl::sycl::id<1> i) {
            acc[i] = 1.0f;
        });
    }).wait();
}

Tensor* Tensor::reshape_tensor(std::vector<int>& new_shape, int new_ndim){
    ndim = new_ndim;
    for(int i = 0; i < ndim; i++)
        shape[i] = new_shape[i];

    int new_size = 1;
    for (int i = 0; i < new_ndim; i++)
        new_size *= shape[i];

    if (new_size != size) {
        std::cout << "Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor." << std::endl;
        exit(1);
    }

    strides.resize(ndim);
    int stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return this;
}


void Tensor::to_device(const std::string& target_device) {
    cl::sycl::queue queue(cl::sycl::gpu_selector_v);
    if (target_device == "sycl" && device == "cpu") {
        cpu_to_sycl(*this, queue);
    } else if (target_device == "cpu" &&  device == "sycl") {
        sycl_to_cpu(*this, queue);
    }
}

void printRecursive(const float* data, const std::vector<int>& shape, int depth, int index, std::ostream& os) {
    if (depth == shape.size() - 1) {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            os << data[index++] << (i < shape[depth] - 1 ? ", " : "");
        }
        os << "]" << (index < shape[depth] ? ", " : "");
    } else {
        os << "[";
        for (int i = 0; i < shape[depth]; ++i) {
            printRecursive(data, shape, depth + 1, index, os);
            index += shape[depth + 1];
        }
        os << "]" << (index < shape[depth] * shape[depth + 1] ? ", " : "");
    }
}

void Tensor::print() const {
    std::cout << "Tensor(data=";

    if (shape.empty()) {
        std::cerr << "Tensor shape is empty." << std::endl;
        return;
    }

    printRecursive(data.get(), shape, 0, 0, std::cout);

    std::cout << ", shape=[";
    for (int i = 0; i < ndim; ++i) {
        std::cout << shape[i] << (i < ndim - 1 ? ", " : "");
    }
    std::cout << "], device=" << device << ")\n";
}

void cpu_to_sycl(Tensor& tensor, cl::sycl::queue& q) {
    if (tensor.device == "sycl") return;

    tensor.data_gpu = cl::sycl::buffer<float, 1>(tensor.data.get(), cl::sycl::range<1>(tensor.size));
    tensor.device = "sycl";

}

void sycl_to_cpu(Tensor& tensor, cl::sycl::queue& q) {
    if (tensor.device == "cpu") return;

    auto data_host = std::make_unique<float[]>(tensor.size);
    q.submit([&](cl::sycl::handler& cgh) {
        auto acc = tensor.data_gpu->get_access<cl::sycl::access::mode::read>(cgh);
        cgh.copy(acc, data_host.get());
    }).wait();

    tensor.data = std::move(data_host);
    tensor.device = "cpu";

}

void Tensor::transpose_1D_tensor_cpu() {
    auto data_host = std::make_unique<float[]>(size);

    for (int i = 0; i < shape[0]; i++)
        data_host[i] = data[i];

    data = std::move(data_host);
}

void Tensor::transpose_2D_tensor_cpu() {
    int rows = shape[0];
    int cols = shape[1];

    auto data_host = std::make_unique<float[]>(size);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data_host[j * rows + i] = data[i * cols + j];

    data = std::move(data_host);
}

void Tensor::transpose_3D_tensor_cpu() {
    int depth = shape[0];
    int rows = shape[1];
    int cols = shape[2];

    auto data_host = std::make_unique<float[]>(size);

    for (int i = 0; i < depth; i++)
        for (int j = 0; j < rows; j++)
            for (int k = 0; k < cols; k++)
                data_host[k * rows * depth + j * depth + i] = data[i * rows * cols + j * cols + k];

    data = std::move(data_host);
}

void Tensor::transpose_1D_tensor_sycl() {
    sycl::queue queue(sycl::gpu_selector_v);

    sycl::buffer<float, 1> data_buffer(data.get(), sycl::range<1>(size));
    auto data_host = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> data_host_buffer(data_host.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_host_acc = data_host_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class transpose_1d>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            data_host_acc[idx] = data_acc[idx];
        });
    }).wait();

    data = std::move(data_host);
}

void Tensor::transpose_2D_tensor_sycl() {
    int rows = shape[0];
    int cols = shape[1];

    sycl::queue queue(sycl::gpu_selector_v);

    sycl::buffer<float, 1> data_buffer(data.get(), sycl::range<1>(size));
    auto data_host = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> data_host_buffer(data_host.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_host_acc = data_host_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class transpose_2d>(sycl::range<2>(rows, cols), [=](sycl::item<2> item) {
            int i = item.get_id(0);
            int j = item.get_id(1);
            data_host_acc[j * rows + i] = data_acc[i * cols + j];
        });
    }).wait();

    shape = {cols, rows};
    data = std::move(data_host);
}

void Tensor::transpose_3D_tensor_sycl() {
    int depth = shape[0];
    int rows = shape[1];
    int cols = shape[2];

    sycl::queue queue(sycl::gpu_selector_v);

    sycl::buffer<float, 1> data_buffer(data.get(), sycl::range<1>(size));
    auto data_host = std::make_unique<float[]>(size);
    sycl::buffer<float, 1> data_host_buffer(data_host.get(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto data_acc = data_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_host_acc = data_host_buffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class transpose_3d>(sycl::range<3>(depth, rows, cols), [=](sycl::item<3> item) {
            int i = item.get_id(0);
            int j = item.get_id(1);
            int k = item.get_id(2);
            data_host_acc[k * rows * depth + j * depth + i] = data_acc[i * rows * cols + j * cols + k];
        });
    }).wait();

    shape = {cols, rows, depth};
    data = std::move(data_host);
}

Tensor* Tensor::transpose_axes_tensor(int axis1, int axis2) {
    if (axis1 < 0 || axis1 >= ndim || axis2 < 0 || axis2 >= ndim) {
        std::cerr << "Invalid axes for transposition." << std::endl;
        return nullptr;
    }

    std::vector<int> new_shape = shape;
    std::swap(new_shape[axis1], new_shape[axis2]);

    std::vector<int> new_strides = strides;
    std::swap(new_strides[axis1], new_strides[axis2]);

    std::unique_ptr<float[]> new_data = std::make_unique<float[]>(size);

    if (device == "cpu") {
        for (int i = 0; i < size; ++i) {
            std::vector<int> indices(ndim);
            int idx = i;
            for (int j = ndim - 1; j >= 0; --j) {
                indices[j] = idx % shape[j];
                idx /= shape[j];
            }
            std::swap(indices[axis1], indices[axis2]);
            int new_idx = 0;
            for (int j = 0; j < ndim; ++j) {
                new_idx += indices[j] * new_strides[j];
            }
            new_data[new_idx] = data[i];
        }

        Tensor* result = new Tensor(std::move(new_data), new_shape, device);
        return result;
    } else if (device == "sycl") {
        if (!data_gpu) {
            std::cerr << "SYCL buffer must be initialized before using it." << std::endl;
            return nullptr;
        }

        try {
            sycl::queue queue(sycl::gpu_selector_v);

            sycl::buffer<float, 1> new_data_buffer(new_data.get(), sycl::range<1>(size));
            sycl::buffer<float, 1> data_buffer(data.get(), sycl::range<1>(size));
            sycl::buffer<int, 1> shape_buffer(shape.data(), sycl::range<1>(ndim));
            sycl::buffer<int, 1> new_strides_buffer(new_strides.data(), sycl::range<1>(ndim));

            queue.submit([&](sycl::handler& cgh) {
                auto data_acc = data_buffer.get_access<sycl::access::mode::read>(cgh);
                auto new_data_acc = new_data_buffer.get_access<sycl::access::mode::write>(cgh);
                auto shape_acc = shape_buffer.get_access<sycl::access::mode::read>(cgh);
                auto new_strides_acc = new_strides_buffer.get_access<sycl::access::mode::read>(cgh);

                cgh.parallel_for<class transpose_tensor>(sycl::range<1>(size), [=](sycl::id<1> idx) {
                    int index = idx[0];
                    int remainder = index;
                    int ndim = shape_acc.get_range()[0];

                    int indices[10];
                    for (int j = ndim - 1; j >= 0; --j) {
                        indices[j] = remainder % shape_acc[j];
                        remainder /= shape_acc[j];
                    }
                    std::swap(indices[axis1], indices[axis2]);

                    int new_index = 0;
                    for (int j = 0; j < ndim; ++j) {
                        new_index += indices[j] * new_strides_acc[j];
                    }

                    new_data_acc[new_index] = data_acc[index];
                });
            }).wait();

            Tensor* result = new Tensor(std::move(new_data), new_shape, device);
            result->data_gpu = std::make_optional<sycl::buffer<float, 1>>(new_data_buffer);

            return result;
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            return nullptr;
        }
    } else {
        std::cerr << "Unsupported device type." << std::endl;
        return nullptr;
    }
}
