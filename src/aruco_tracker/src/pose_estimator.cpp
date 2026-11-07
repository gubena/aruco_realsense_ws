#include "aruco_tracker/pose_estimator.hpp"

#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace aruco_tracker
{
namespace
{

inline double rmsReprojectionError(
  const std::vector<cv::Point2f> & observed, const std::vector<cv::Point2f> & projected)
{
  if (observed.size() != projected.size() || observed.empty()) {
    return 0.0;
  }

  double sum = 0.0;
  for (size_t i = 0; i < observed.size(); ++i) {
    const cv::Point2f diff = observed[i] - projected[i];
    sum += diff.x * diff.x + diff.y * diff.y;
  }
  return std::sqrt(sum / static_cast<double>(observed.size()));
}

}  // namespace

PoseEstimator::PoseEstimator(
  cv::Mat camera_matrix, cv::Mat dist_coeffs, double marker_size_m,
  double max_reprojection_error_px)
: camera_matrix_(camera_matrix.clone()),
  dist_coeffs_(dist_coeffs.clone()),
  marker_size_m_(marker_size_m),
  max_reprojection_error_px_(max_reprojection_error_px)
{
  if (!(marker_size_m_ > 0.0)) {
    throw std::invalid_argument("marker_size_m must be > 0");
  }
  if (camera_matrix_.empty() || camera_matrix_.rows != 3 || camera_matrix_.cols != 3) {
    throw std::invalid_argument("camera_matrix must be a 3x3 matrix");
  }
  if (camera_matrix_.type() != CV_64F && camera_matrix_.type() != CV_32F) {
    throw std::invalid_argument("camera_matrix must be floating point");
  }
  if (dist_coeffs_.empty()) {
    throw std::invalid_argument("dist_coeffs must not be empty");
  }
  if (dist_coeffs_.rows != 1 && dist_coeffs_.cols != 1 && dist_coeffs_.rows != 1 && dist_coeffs_.cols > 0) {
    throw std::invalid_argument("dist_coeffs must be a 1xn row or column vector");
  }
  if (!(max_reprojection_error_px_ >= 0.0)) {
    throw std::invalid_argument("max_reprojection_error_px must be >= 0");
  }

  const double s = marker_size_m_ * 0.5;
  object_points_ = {
    cv::Point3f(-s, s, 0.0),
    cv::Point3f(s, s, 0.0),
    cv::Point3f(s, -s, 0.0),
    cv::Point3f(-s, -s, 0.0),
  };
}

std::vector<MarkerPose> PoseEstimator::estimate(const std::vector<MarkerDetection> & detections)
{
  std::vector<MarkerPose> poses;
  poses.reserve(detections.size());

  for (const auto & detection : detections) {
    image_points_.assign(detection.corners.begin(), detection.corners.end());
    cv::solvePnP(
      object_points_, image_points_, camera_matrix_, dist_coeffs_, rvec_, tvec_,
      false, cv::SOLVEPNP_IPPE_SQUARE);

    cv::projectPoints(
      object_points_, rvec_, tvec_, camera_matrix_, dist_coeffs_, projected_points_);
    const double reproj_error = rmsReprojectionError(image_points_, projected_points_);

    if (reproj_error > max_reprojection_error_px_ || tvec_.at<double>(2) <= 0.0) {
      continue;
    }

    MarkerPose pose;
    pose.id = detection.id;
    pose.rvec = cv::Vec3d(rvec_.at<double>(0), rvec_.at<double>(1), rvec_.at<double>(2));
    pose.tvec = cv::Vec3d(tvec_.at<double>(0), tvec_.at<double>(1), tvec_.at<double>(2));
    pose.reprojection_error_px = reproj_error;
    poses.push_back(pose);
  }

  return poses;
}

}  // namespace aruco_tracker
