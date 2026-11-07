#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

namespace aruco_tracker
{

struct CameraIntrinsics
{
  cv::Mat camera_matrix;        ///< 3x3, CV_64F
  cv::Mat dist_coeffs;          ///< 1xN, CV_64F (N = 4, 5, 8, ...)
  cv::Size image_size;          ///< calibration resolution; 0x0 if the file does not say
  std::string distortion_model; ///< e.g. "plumb_bob", "rational_polynomial", "equidistant"
  std::string camera_name;
};

/// Loads the file written by ROS' camera_calibration (ost.yaml / <camera>.yaml):
///
///   image_width: 1920
///   image_height: 1080
///   camera_name: my_cam
///   camera_matrix:            {rows: 3, cols: 3, data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]}
///   distortion_model: plumb_bob
///   distortion_coefficients:  {rows: 1, cols: 5, data: [k1, k2, p1, p2, k3]}
///
/// (block or flow style both work; rectification/projection matrices are ignored).
/// Throws std::runtime_error with a readable message on any problem.
inline CameraIntrinsics loadCameraIntrinsics(const std::string & path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("Cannot read camera intrinsics '" + path + "': " + e.what());
  }

  auto readData = [&](const char * key, size_t min_count, size_t max_count) {
      const YAML::Node node = root[key];
      if (!node || !node["data"]) {
        throw std::runtime_error(
                "Camera intrinsics '" + path + "': missing '" + key + ": data: [...]'");
      }
      std::vector<double> values;
      try {
        values = node["data"].as<std::vector<double>>();
      } catch (const YAML::Exception & e) {
        throw std::runtime_error(
                "Camera intrinsics '" + path + "': bad '" + key + "' data: " + e.what());
      }
      if (values.size() < min_count || values.size() > max_count) {
        throw std::runtime_error(
                "Camera intrinsics '" + path + "': '" + key + "' has " +
                std::to_string(values.size()) + " values, unexpected");
      }
      return values;
    };

  CameraIntrinsics out;

  const std::vector<double> k = readData("camera_matrix", 9, 9);
  out.camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(k.data())).clone();

  const std::vector<double> d = readData("distortion_coefficients", 4, 14);
  out.dist_coeffs = cv::Mat(1, static_cast<int>(d.size()), CV_64F, const_cast<double *>(d.data()))
    .clone();

  if (root["image_width"] && root["image_height"]) {
    out.image_size = cv::Size(root["image_width"].as<int>(), root["image_height"].as<int>());
  }
  if (root["distortion_model"]) {
    out.distortion_model = root["distortion_model"].as<std::string>();
  }
  if (root["camera_name"]) {
    out.camera_name = root["camera_name"].as<std::string>();
  }
  return out;
}

}  // namespace aruco_tracker
