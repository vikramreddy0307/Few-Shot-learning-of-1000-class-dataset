#include "CNNEncoder.h"

std::tuple<torch::nn::Module,torch::nn::Module,torch::nn::Module,torch::nn::Module,torch::nn::Module> make_features(std::vector<int> &cfg, bool batch_norm){
    torch::nn::Sequential features;
    int v=64;
    features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v).eps(1e-05).momentum(0.1).affine(True).track_running_stats(true))));
    features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    int in_channels = v;
    int cnt=2;
    for(auto v : cfg){
        if(v==-1){
            cnt+=1;
            features->push_back(torch::nn::MaxPool2d(maxpool_options(2,2,0,1)));
        }
        else{
            auto conv2d = torch::nn::Conv2d(conv_options(in_channels,v,3,1,1));
            features->push_back(conv2d)
            
            if(batch_norm){
                features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v).eps(1e-05).momentum(0.1).affine(True).track_running_stats(true))));
            }
            features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            cnt+=3;
            in_channels = v;
            if (cnt==5)
            {
                features_5=register_module("features",features_);
                torch::nn::Sequential features;
            }
            else if(cnt==12)
            {
                features_12=register_module("features",features_);
                torch::nn::Sequential features;
            }
            else if(cnt==22)
            {
                features_22=register_module("features",features_);
                torch::nn::Sequential features;
            }
            else if(cnt==32)
            {
                features_32=register_module("features",features_);
                torch::nn::Sequential features;
            }
            else if(cnt==42)
            {
                features_42=register_module("features",features_);
                torch::nn::Sequential features;
            }
            else{}
        }
    }
    return std::make_tuple(features_4,features_12,features_22,features_32,features_42);
}

VGGImpl::VGGImpl(std::vector<int> &cfg, int num_classes, bool batch_norm){
    std::vector<int> cfg_dd = {64,  -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto [features_4,features_11,features_21,features_31,features_41 = make_features(cfg,batch_norm,0,5);
    
    // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    // classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    // classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    // classifier->push_back(torch::nn::Dropout());
    // classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    // classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    // classifier->push_back(torch::nn::Dropout());
    // classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

    // features_ = register_module("features",features_);
    // classifier = register_module("classifier",classifier);
}

std::tuple<torch::Tensor,torch::Tensor> VGGImpl::forward(torch::Tensor x){
    x_4 = features_4->forward(x);
    x_11 = features_4->forward(x_4);
    x_21 = features_4->forward(x_11);
    x_31 = features_4->forward(x_21);
    x_41 = features_4->forward(x_31);
    torch::Tensor feature_list=torch::cat({ x_4, x_11,x_21,x_31,x_41 }, 0);
    // x = avgpool(x);
    // x = torch::flatten(x,1);
    // x = classifier->forward(x);
    return std::make_tuple(x_41,feature_list)
}
