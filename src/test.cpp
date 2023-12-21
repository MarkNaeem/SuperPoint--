void test()
{
    // Example data to test the functions
    // Adjust the tensor sizes and values according to the requirements of each function

    // Testing simple_nms
    torch::Tensor scores = torch::rand({1, 1, 10, 10}); // Example scores tensor
    int nms_radius = 4;
    auto nms_result = simple_nms(scores, nms_radius);
    std::cout << "NMS Result: " << nms_result << std::endl;

    // Testing remove_borders
    torch::Tensor keypoints = torch::rand({10, 2}) * 100; // Example keypoints tensor
    torch::Tensor kp_scores = torch::rand({10});          // Example scores for keypoints
    int border = 10, height = 100, width = 100;
    auto [keypoints_filtered, scores_filtered] = remove_borders(keypoints, kp_scores, border, height, width);
    std::cout << "Filtered Keypoints: " << keypoints_filtered << std::endl;

    // Testing top_k_keypoints
    int k = 5;
    auto [top_k_kp, top_k_scores] = top_k_keypoints(keypoints_filtered, scores_filtered, k);
    std::cout << "Top K Keypoints: " << top_k_kp << std::endl;

    // Testing sample_descriptors
    torch::Tensor descriptors = torch::rand({1, 256, 10, 10}); // Example descriptors tensor
    auto sampled_descriptors = sample_descriptors(top_k_kp, descriptors);
    std::cout << "Sampled Descriptors: " << sampled_descriptors << std::endl;
}
