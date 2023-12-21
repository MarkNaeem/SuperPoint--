#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

#include <fstream>
#include <iomanip>

#include "tools.h"

typedef Eigen::Matrix<double, 4, 4> Matrix4d;

namespace fs = std::filesystem;

int countImagesInDirectory(const std::string &directory)
{
    int count = 0;
    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file())
        {
            // Check if the file is an image (by extension)
            std::string extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                count++;
            }
        }
    }
    return count;
}

cv::Mat readKittiCalibration(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.substr(0, 2) == "P2")
        {
            std::istringstream iss(line.substr(3)); // Skip the 'P2'

            std::vector<double> values;
            double value;
            while (iss >> value)
            {
                values.push_back(value);
            }

            cv::Mat K = (cv::Mat_<double>(3, 3) << values[0], values[1], values[2],
                         values[4], values[5], values[6],
                         values[8], values[9], values[10]);

            return K;
        }
    }

    throw std::runtime_error("Intrinsic matrix not found in file: " + file_path);
}

bool estimatePose(const torch::Tensor &keypoints1,
                  const torch::Tensor &keypoints2,
                  const cv::Mat &K, // Camera intrinsic matrix
                  Matrix4d &current_pose)
{
    // Convert keypoints to point2f
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < keypoints1.size(0); i++)
    {
        points1.push_back(cv::Point2f(std::lround(keypoints1[i][0].item<float>()), std::lround(keypoints1[i][1].item<float>())));
        points2.push_back(cv::Point2f(std::lround(keypoints2[i][0].item<float>()), std::lround(keypoints2[i][1].item<float>())));
    }

    // Find Essential Matrix
    cv::Mat inliers;
    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0, inliers);

    if (E.empty())
    {
        std::cerr << "Essential matrix estimation failed." << std::endl;
        return false;
    }

    // Recover pose from Essential Matrix
    cv::Mat R, t;
    Matrix4d transformation_matrix = Matrix4d::Identity();

    cv::recoverPose(E, points1, points2, K, R, t, inliers);

    // Use intermediate Eigen matrices for conversion
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;

    // Convert from cv::Mat to Eigen matrices
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(t, t_eigen);

    // Assign to blocks in the transformation matrix
    transformation_matrix.block<3, 3>(0, 0) = R_eigen;
    transformation_matrix.block<3, 1>(0, 3) = t_eigen;

    // Update the current pose
    current_pose = current_pose * transformation_matrix;

    return true;
}

std::string getImagePath(const std::string &base_folder, int image_index)
{
    std::ostringstream path;
    path << base_folder << std::setw(6) << std::setfill('0') << image_index << ".png";
    return path.str();
}

void saveTrajectoryToFile(const std::vector<Matrix4d> &trajectory, const std::string &file_path)
{
    std::ofstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file for writing: " << file_path << std::endl;
        return;
    }

    for (const auto &pose : trajectory)
    {
        for (int i = 0; i < 3; ++i)
        { // Only the first three rows
            for (int j = 0; j < 4; ++j)
            { // Only the first four columns
                file << std::setprecision(9) << pose(i, j);
                if (j < 3)
                    file << " ";
            }
            file << std::endl;
        }
    }

    file.close();
}

int main(int argc, const char *argv[])
{
    if (argc <= 2)
    {
        std::cerr << "Usage:" << std::endl;
        std::cerr << argv[0] << " <downscaled_width> <datatrack>" << std::endl;
        return 1;
    }

    torch::manual_seed(1);
    torch::autograd::GradMode::set_enabled(false);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    std::string data_track = argv[2];
    std::string base_folder = "/home/marknaeem97/dataset/sequences/" + data_track + "/image_0/";

    // Load the number of images in the folder
    std::cout << "counting images" << std::endl;
    int num_images = countImagesInDirectory(base_folder);

    // Initialize variables
    Matrix4d current_pose = Matrix4d::Identity();
    std::vector<Matrix4d> trajectory;
    trajectory.push_back(current_pose);

    torch::Tensor keypoints0, scores0, descriptors0;
    torch::Tensor keypoints1, scores1, descriptors1;
    torch::Dict<std::string, torch::Tensor> pred;

    cv::Mat K = readKittiCalibration("/home/marknaeem97/dataset/sequences/" + data_track + "/calib.txt");

    int target_width = std::stoi(argv[1]);

    // Look for the TorchScript module files in the executable directory
    auto executable_dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
    auto module_path = executable_dir / "SuperPoint.zip";
    if (!std::filesystem::exists(module_path))
    {
        std::cerr << "Could not find the TorchScript module file " << module_path << std::endl;
        return 1;
    }
    torch::jit::script::Module superpoint = torch::jit::load(module_path);
    superpoint.eval();
    superpoint.to(device);

    module_path = executable_dir / "SuperGlue.zip";
    if (!std::filesystem::exists(module_path))
    {
        std::cerr << "Could not find the TorchScript module file " << module_path << std::endl;
        return 1;
    }
    torch::jit::script::Module superglue = torch::jit::load(module_path);
    superglue.eval();
    superglue.to(device);

    // Load the first image using OpenCV
    std::cout << "first image" << std::endl;
    std::string first_image_path = base_folder + "000000.png";
    torch::Tensor prev_frame = read_image(first_image_path, target_width).to(device);

    std::tie(keypoints0, scores0, descriptors0) = unpack_result(superpoint.forward({prev_frame}));

    try
    {
        // Main loop for processing images
        for (int i = 1; i < num_images; ++i)
        {
            std::cout << i << " out of " << num_images << std::endl;
            // Load current image
            std::string image_path = getImagePath(base_folder, i);
            torch::Tensor cur_frame = read_image(image_path, target_width).to(device);

            std::tie(keypoints1, scores1, descriptors1) = unpack_result(superpoint.forward({cur_frame}));

            torch::Dict<std::string, torch::Tensor> input;

            input.insert("image0", prev_frame);
            input.insert("image1", cur_frame);
            input.insert("keypoints0", keypoints0.unsqueeze(0));
            input.insert("keypoints1", keypoints1.unsqueeze(0));
            input.insert("scores0", scores0.unsqueeze(0));
            input.insert("scores1", scores1.unsqueeze(0));
            input.insert("descriptors0", descriptors0.unsqueeze(0));
            input.insert("descriptors1", descriptors1.unsqueeze(0));
            pred = toTensorDict(superglue.forward({input}));

            auto matches = pred.at("matches0")[0];
            auto valid = torch::nonzero(matches > -1).squeeze();
            auto mkpts0 = keypoints0.index_select(0, valid);
            auto mkpts1 = keypoints1.index_select(0, matches.index_select(0, valid));
            auto confidence = pred.at("matching_scores0")[0].index_select(0, valid);

            if (!estimatePose(mkpts0, mkpts1, K, current_pose))
            {
                return -1; // Pose estimation failed
            }

            trajectory.push_back(current_pose);

            keypoints0 = keypoints1;
            scores0 = scores1;
            descriptors0 = descriptors1;
            prev_frame = cur_frame;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    saveTrajectoryToFile(trajectory, "/home/marknaeem97/output_trajectory.txt");

    return 0;
}
