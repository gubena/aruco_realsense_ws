#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace aruco_tracker
{

struct MarkerDetectorConfig
{
  /// OpenCV predefined dictionary. Must match the dictionary the markers were printed from.
  std::string dictionary{"DICT_4X4_50"};

  /// (0, 1]. Candidates are searched on a copy of the image scaled by this factor, which
  /// makes detection roughly 1/scale^2 faster. Corners are always reported in full-resolution
  /// pixel coordinates.
  double detection_scale{1.0};

  /// Refine the 4 corners of every marker with cv::cornerSubPix on the full-resolution image.
  bool subpixel_refine{true};

  /// Minimum marker perimeter relative to max(image width, image height), measured on the
  /// image that is searched. Lower values detect smaller / farther markers but are slower.
  double min_marker_perimeter_rate{0.03};

  /// Maximum marker perimeter relative to max(image width, image height).
  double max_marker_perimeter_rate{4.0};

  /// Accuracy of polygon approximation for marker contour extraction.
  double polygonal_approx_accuracy_rate{0.05};

  /// Minimum distance between different markers in the same image, relative to the marker size.
  double min_marker_distance_rate{0.05};

  /// Minimum distance from a marker corner to the image border, in pixels.
  int min_distance_to_border{3};

  /// Minimum corner distance rate used by the ArUco detector.
  double min_corner_distance_rate{0.05};

  /// Error correction rate relative to the dictionary's maximum correction capability.
  double error_correction_rate{0.6};

  /// Maximum allowed erroneous bits in the marker border, relative to border size.
  double max_erroneous_bits_in_border_rate{0.35};

  /// Minimum Otsu standard deviation required for a marker candidate; rejects low-contrast images.
  double min_otsu_std_dev{5.0};

  /// Corner refinement mode (0 = none, 1 = subpixel, 2 = contour).
  int corner_refinement_method{1};

  int adaptive_thresh_constant{7};
  int adaptive_thresh_win_size_min{3};
  int adaptive_thresh_win_size_max{23};
  int adaptive_thresh_win_size_step{10};
};

struct MarkerDetection
{
  int id{-1};

  /// Corners of the black square (the white quiet zone is NOT part of it), in full-resolution
  /// pixel coordinates, in the marker's own order, clockwise:
  ///   0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
  /// "top-left" is defined by the marker's pattern, not by the image.
  std::array<cv::Point2f, 4> corners;
};

/// Thin wrapper around OpenCV's ArUco detector. Hides the API change that happened in
/// OpenCV 4.7 (cv::aruco::detectMarkers -> cv::aruco::ArucoDetector), so the rest of the
/// code builds on Humble/Jazzy (OpenCV 4.5/4.6) and on newer OpenCV alike.
class MarkerDetector
{
public:
  /// Throws std::invalid_argument on a bad configuration.
  explicit MarkerDetector(const MarkerDetectorConfig & config);
  ~MarkerDetector();

  MarkerDetector(const MarkerDetector &) = delete;
  MarkerDetector & operator=(const MarkerDetector &) = delete;

  /// @param gray  CV_8UC1 image. May be a non-continuous view (e.g. wrapping a ROS message).
  /// @param rejected_count  Optional output count of candidate markers rejected by the detector.
  std::vector<MarkerDetection> detect(const cv::Mat & gray, size_t * rejected_count = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Draw outlines (corner 0 marked with a red dot) and ids on a BGR image.
/// @param scale  factor from full-resolution pixel coordinates to `bgr` pixel coordinates.
void drawDetections(
  cv::Mat & bgr, const std::vector<MarkerDetection> & detections, double scale = 1.0);

}  // namespace aruco_tracker
