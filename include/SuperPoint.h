#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <torch/torch.h>
#include <unordered_map>

class SuperPoint : public torch::nn::Module
{
public:
    explicit SuperPoint(const std::unordered_map<std::string, int> &config);
    torch::Tensor forward(torch::Tensor x);

private:
    std::unordered_map<std::string, int> config;
    torch::nn::Conv2d conv1a{nullptr}, conv1b{nullptr}, conv2a{nullptr}, conv2b{nullptr},
        conv3a{nullptr}, conv3b{nullptr}, conv4a{nullptr}, conv4b{nullptr},
        convPa{nullptr}, convPb{nullptr}, convDa{nullptr}, convDb{nullptr};
    torch::nn::ReLU relu;
    torch::nn::MaxPool2d pool{nullptr};
};
#endif // SUPERPOINT_H
