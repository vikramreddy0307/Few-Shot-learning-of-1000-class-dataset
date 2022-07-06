#pragma once

#include<torch/torch.h>

/*
class CNNEncoder : public torch::nn::Module
{
    public:
    explicit CNNEncoder();

    private:
    torch::nn::Sequential layer1{
        // torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 63, 3).stride(1).padding(1))
        
    };
    torch::nn::Sequential layer2{
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(0).dilation(1))

        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 63, 3).stride(1).padding(1))
        
    };

    torch::nn::Sequential layer3{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };
    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer3{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
    };

     torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };
    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer3{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
    };

    torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({4, 4})};
    torch::nn::Linear fc;

}
*/

#ifndef CNNENCODER_H
#define CNNENCODER_H
#undef slots
#include<torch/script.h>
#include<torch/torch.h>
#define slots Q_SLOTS
#include<vector>

inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;
}

inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride,int padding,int dilation ){
    torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
    maxpool_options.stride(stride);
    maxpool_options.padding(padding);
    maxpool_options.dilation(dilation);
    return maxpool_options;
}

// inline torch::nn::MaxPool2dOptions(int size)
// {
//     torch::nn::BatchNorm2dOptions  batchnorm_options=torch::nn::BatchNorm2dOptions(size);
// }
// torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)

std::tuple<torch::nn::Module,torch::nn::Module,torch::nn::Module,torch::nn::Module,torch::nn::Module> make_features(std::vector<int> &cfg, bool batch_norm);

class VGGImpl: public torch::nn::Module
{
private:
    torch::nn::Sequential features_{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier;
public:
    VGGImpl(std::vector<int> &cfg, int num_classes = 1000, bool batch_norm = false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CNNENcoder);

#endif // VGG_H
