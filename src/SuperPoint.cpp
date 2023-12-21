#include "SuperPoint.h"
#include "tools.h"

SuperPoint::SuperPoint(const std::unordered_map<std::string, int>& config) : config(config) {
    // Initialize layers
    conv1a = register_module("conv1a", torch::nn::Conv2d(1, 64, 3).padding(1));
    conv1b = register_module("conv1b", torch::nn::Conv2d(64, 64, 3).padding(1));
    conv2a = register_module("conv2a", torch::nn::Conv2d(64, 64, 3).padding(1));
    conv2b = register_module("conv2b", torch::nn::Conv2d(64, 64, 3).padding(1));
    conv3a = register_module("conv3a", torch::nn::Conv2d(64, 128, 3).padding(1));
    conv3b = register_module("conv3b", torch::nn::Conv2d(128, 128, 3).padding(1));
    conv4a = register_module("conv4a", torch::nn::Conv2d(128, 128, 3).padding(1));
    conv4b = register_module("conv4b", torch::nn::Conv2d(128, 128, 3).padding(1));

    convPa = register_module("convPa", torch::nn::Conv2d(128, 256, 3).padding(1));
    convPb = register_module("convPb", torch::nn::Conv2d(256, 65, 1));

    convDa = register_module("convDa", torch::nn::Conv2d(128, 256, 3).padding(1));
    convDb = register_module("convDb", torch::nn::Conv2d(256, config.at("descriptor_dim"), 1));

    pool = register_module("pool", torch::nn::MaxPool2d(2, 2));
    relu = register_module("relu", torch::nn::ReLU());
}

torch::Tensor SuperPoint::forward(torch::Tensor x) {
    // Shared Encoder
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = pool->forward(x);
    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = pool->forward(x);
    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = pool->forward(x);
    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    // Compute the dense keypoint scores
    auto cPa = relu(convPa->forward(x));
    auto scores = convPb->forward(cPa);
    scores = torch::nn::functional::softmax(scores, 1);
    scores = scores.narrow(1, 0, scores.size(1) - 1);
    auto [b, _, h, w] = scores.sizes();
    scores = scores.permute({0, 2, 3, 1}).reshape({b, h, w, 8, 8});
    scores = scores.permute({0, 1, 3, 2, 4}).reshape({b, h * 8, w * 8});
    scores = simple_nms(scores, config["nms_radius"]);

    // Extract keypoints (this is simplified, actual implementation may need additional logic)
    // auto keypoints = ...;

    // Compute the dense descriptors
    auto cDa = relu(convDa->forward(x));
    auto descriptors = convDb->forward(cDa);
    descriptors = torch::nn::functional::normalize(descriptors, 2, 1);

    // Extract descriptors (simplified)
    // auto descriptors = ...;


    return x; 
}
