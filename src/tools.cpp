#include "tools.h"

cv::Mat draw_keypoints(const torch::Tensor &img, const torch::Tensor &keypoints)
{
    cv::Mat out = tensor2mat(img);
    out.convertTo(out, CV_8U, 255.0f);
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < keypoints.size(0); ++i)
    {
        auto kp = keypoints[i];
        cv::Point p(std::roundl(kp[0].item<float>()), std::roundl(kp[1].item<float>()));
        cv::circle(out, p, 2, {0, 0, 255}, -1, cv::LINE_AA);
    }
    return out;
}

cv::Mat make_matching_plot_fast(const torch::Tensor &image0, const torch::Tensor &image1,
                                const torch::Tensor &kpts0, const torch::Tensor &kpts1,
                                const torch::Tensor &mkpts0, const torch::Tensor &mkpts1,
                                const torch::Tensor &confidence, bool show_keypoints,
                                int margin)
{
    cv::Mat imgmat0 = tensor2mat(image0);
    imgmat0.convertTo(imgmat0, CV_8U, 255.0f);
    cv::Mat imgmat1 = tensor2mat(image1);
    imgmat1.convertTo(imgmat1, CV_8U, 255.0f);

    if (show_keypoints)
    {
        const cv::Scalar white(255, 255, 255);
        const cv::Scalar black(0, 0, 0);
        for (int i = 0; i < kpts0.size(0); ++i)
        {
            auto kp = kpts0[i];
            cv::Point pt(std::lround(kp[0].item<float>()), std::lround(kp[1].item<float>()));
            cv::circle(imgmat0, pt, 2, black, -1, cv::LINE_AA);
            cv::circle(imgmat0, pt, 1, white, -1, cv::LINE_AA);
        }
        for (int i = 0; i < kpts1.size(0); ++i)
        {
            auto kp = kpts1[i];
            cv::Point pt(std::lround(kp[0].item<float>()), std::lround(kp[1].item<float>()));
            cv::circle(imgmat1, pt, 2, black, -1, cv::LINE_AA);
            cv::circle(imgmat1, pt, 1, white, -1, cv::LINE_AA);
        }
    }

    int H0 = imgmat0.rows, W0 = imgmat0.cols;
    int H1 = imgmat1.rows, W1 = imgmat1.cols;
    int H = std::max(H0, H1), W = W0 + W1 + margin;

    cv::Mat out = 255 * cv::Mat::ones(H, W, CV_8U);
    imgmat0.copyTo(out.rowRange(0, H0).colRange(0, W0));
    imgmat1.copyTo(out.rowRange(0, H1).colRange(W0 + margin, W));
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

    // Apply colormap to confidences
    cv::Mat conf_mat = tensor2mat(confidence.unsqueeze(0));
    conf_mat.convertTo(conf_mat, CV_8U, 255.0f);
    cv::Mat colors;
    cv::applyColorMap(conf_mat, colors, cv::COLORMAP_JET);

    int n = std::min(mkpts0.size(0), mkpts1.size(0));
    for (int i = 0; i < n; ++i)
    {
        auto kp0 = mkpts0[i];
        auto kp1 = mkpts1[i];
        cv::Point pt0(std::lround(kp0[0].item<float>()), std::lround(kp0[1].item<float>()));
        cv::Point pt1(std::lround(kp1[0].item<float>()), std::lround(kp1[1].item<float>()));
        auto c = colors.at<cv::Vec3b>({i, 0});
        cv::line(out, pt0, {pt1.x + margin + W0, pt1.y}, c, 1, cv::LINE_AA);
        // display line end-points as circles
        cv::circle(out, pt0, 2, c, -1, cv::LINE_AA);
        cv::circle(out, {pt1.x + margin + W0, pt1.y}, 2, c, -1, cv::LINE_AA);
    }

    return out;
}

torch::Tensor read_image(const std::string &path, int target_width)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    int target_height = std::lround((float)target_width / image.cols * image.rows);
    image.convertTo(image, CV_32F, 1.0f / 255.0f);
    cv::resize(image, image, {target_width, target_height});

    torch::Tensor tensor = torch::from_blob(image.data, {1, 1, image.rows, image.cols},
                                            torch::TensorOptions().dtype(torch::kFloat32));
    return tensor.clone();
}

cv::Mat tensor2mat(torch::Tensor tensor)
{
    tensor = tensor.to(torch::kCPU).contiguous();
    cv::Mat mat(tensor.size(-2), tensor.size(-1), CV_32F);
    std::memcpy((void *)mat.data, tensor.data_ptr(), sizeof(float) * tensor.numel());
    return mat;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_result(const torch::IValue &result)
{
    auto dict = result.toGenericDict();
    return {dict.at("keypoints").toTensorVector()[0], //
            dict.at("scores").toTensorVector()[0],    //
            dict.at("descriptors").toTensorVector()[0]};
}

torch::Dict<std::string, torch::Tensor> toTensorDict(const torch::IValue &value)
{
    return c10::impl::toTypedDict<std::string, torch::Tensor>(value.toGenericDict());
}

torch::Tensor max_pool(torch::Tensor x, int nms_radius)
{
    return torch::max_pool2d(x, {nms_radius * 2 + 1, nms_radius * 2 + 1},
                             {1, 1}, {nms_radius, nms_radius});
}

torch::Tensor simple_nms(torch::Tensor scores, int nms_radius)
{
    assert(nms_radius >= 0);

    auto zeros = torch::zeros_like(scores);
    auto max_mask = scores == max_pool(scores, nms_radius);

    for (int i = 0; i < 2; ++i)
    {
        auto supp_mask = max_pool(max_mask.to(torch::kFloat), nms_radius) > 0;
        auto supp_scores = torch::where(supp_mask, zeros, scores);
        auto new_max_mask = supp_scores == max_pool(supp_scores, nms_radius);
        max_mask = max_mask.__ior__(new_max_mask.__iand__(supp_mask.logical_not()));
    }
    return torch::where(max_mask, scores, zeros);
}

std::tuple<torch::Tensor, torch::Tensor> remove_borders(torch::Tensor keypoints, torch::Tensor scores, int border, int height, int width)
{
    auto mask_h = (keypoints.select(1, 0) >= border) & (keypoints.select(1, 0) < (height - border));
    auto mask_w = (keypoints.select(1, 1) >= border) & (keypoints.select(1, 1) < (width - border));
    auto mask = mask_h & mask_w;

    return std::make_tuple(keypoints.index_select(0, mask.nonzero().squeeze(1)), scores.index_select(0, mask.nonzero().squeeze(1)));
}

std::pair<torch::Tensor, torch::Tensor> top_k_keypoints(torch::Tensor keypoints, torch::Tensor scores, int k)
{
    if (k >= keypoints.size(0))
    {
        return std::make_pair(keypoints, scores);
    }

    torch::Tensor values, indices;
    std::tie(values, indices) = torch::topk(scores, k, 0);

    return std::make_pair(keypoints.index_select(0, indices), values);
}

torch::Tensor sample_descriptors(torch::Tensor keypoints, torch::Tensor descriptors, int s)
{
    auto b = descriptors.size(0);
    auto c = descriptors.size(1);
    auto h = descriptors.size(2);
    auto w = descriptors.size(3);

    keypoints = keypoints - s / 2 + 0.5;
    keypoints /= torch::tensor({w * s - s / 2 - 0.5, h * s - s / 2 - 0.5}).to(keypoints).unsqueeze(0);
    keypoints = keypoints * 2 - 1; // normalize to (-1, 1)

    auto options = torch::nn::functional::GridSampleFuncOptions().align_corners(true);
    auto descriptors_sampled = torch::nn::functional::grid_sample(descriptors, keypoints.view({b, 1, -1, 2}), options);

    auto norm_options = torch::nn::functional::NormalizeFuncOptions().p(2).dim(1);
    descriptors_sampled = torch::nn::functional::normalize(descriptors_sampled.view({b, c, -1}), norm_options);

    return descriptors_sampled;
}
