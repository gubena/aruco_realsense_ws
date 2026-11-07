#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "aruco_tracker/marker_detector.hpp"

namespace aruco_tracker
{

struct MarkerPose
{
  int id{-1};
  cv::Vec3d rvec{0.0, 0.0, 0.0};
  cv::Vec3d tvec{0.0, 0.0, 0.0};
  double reprojection_error_px{0.0};
};

class PoseEstimator
{
public:
  PoseEstimator(
    cv::Mat camera_matrix, cv::Mat dist_coeffs, double marker_size_m,
    double max_reprojection_error_px);

  std::vector<MarkerPose> estimate(const std::vector<MarkerDetection> & detections);

private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  double marker_size_m_{0.05};
  double max_reprojection_error_px_{3.0};

  std::vector<cv::Point3f> object_points_;
  std::vector<cv::Point2f> image_points_;
  std::vector<cv::Point2f> projected_points_;
  cv::Mat rvec_;
  cv::Mat tvec_;
};

}  // namespace aruco_tracker
