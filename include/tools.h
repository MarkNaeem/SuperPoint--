#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <tuple>
#include <chrono>
#include <filesystem>
#include <utility>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat draw_keypoints(const torch::Tensor &img, const torch::Tensor &keypoints);

cv::Mat make_matching_plot_fast(const torch::Tensor &image0, const torch::Tensor &image1,
                                const torch::Tensor &kpts0, const torch::Tensor &kpts1,
                                const torch::Tensor &mkpts0, const torch::Tensor &mkpts1,
                                const torch::Tensor &confidence, bool show_keypoints = true,
                                int margin = 10);

torch::Tensor read_image(const std::string &path, int target_width);

cv::Mat tensor2mat(torch::Tensor tensor);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_result(const torch::IValue &result);

torch::Dict<std::string, torch::Tensor> toTensorDict(const torch::IValue &value);

torch::Tensor max_pool(torch::Tensor x, int nms_radius);

torch::Tensor simple_nms(torch::Tensor scores, int nms_radius);

std::tuple<torch::Tensor, torch::Tensor> remove_borders(torch::Tensor keypoints, torch::Tensor scores, int border, int height, int width);

std::pair<torch::Tensor, torch::Tensor> top_k_keypoints(torch::Tensor keypoints, torch::Tensor scores, int k);

torch::Tensor sample_descriptors(torch::Tensor keypoints, torch::Tensor descriptors, int s = 8);

#endif // TOOLS_H
