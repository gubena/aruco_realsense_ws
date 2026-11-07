#include "aruco_tracker/marker_detector.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>

// OpenCV 4.7 moved ArUco into objdetect and replaced the free function detectMarkers()
// by the ArucoDetector class. Ubuntu 22.04 / 24.04 (ROS 2 Humble / Jazzy) ship 4.5 / 4.6.
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#define ARUCO_TRACKER_NEW_ARUCO_API 1
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif

namespace aruco_tracker
{
namespace
{

int predefinedDictionaryId(const std::string & name)
{
  static const std::map<std::string, int> table = {
    {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
    {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
  };
  const auto it = table.find(name);
  if (it == table.end()) {
    throw std::invalid_argument(
            "Unsupported ArUco dictionary '" + name +
            "' (use DICT_4X4_50, DICT_4X4_100, DICT_4X4_250 or DICT_4X4_1000)");
  }
  return it->second;
}

// Works for both DetectorParameters (4.7+, value type) and *Ptr<DetectorParameters> (older).
template<typename Params>
void applyConfig(Params & p, const MarkerDetectorConfig & c)
{
  p.minMarkerPerimeterRate = c.min_marker_perimeter_rate;
  p.maxMarkerPerimeterRate = c.max_marker_perimeter_rate;
  p.polygonalApproxAccuracyRate = c.polygonal_approx_accuracy_rate;
  p.minMarkerDistanceRate = c.min_marker_distance_rate;
  p.minDistanceToBorder = c.min_distance_to_border;
  p.minCornerDistanceRate = c.min_corner_distance_rate;
  p.errorCorrectionRate = c.error_correction_rate;
  p.maxErroneousBitsInBorderRate = c.max_erroneous_bits_in_border_rate;
  p.minOtsuStdDev = c.min_otsu_std_dev;
  p.cornerRefinementMethod = static_cast<decltype(p.cornerRefinementMethod)>(c.corner_refinement_method);
  p.adaptiveThreshConstant = c.adaptive_thresh_constant;
  p.adaptiveThreshWinSizeMin = c.adaptive_thresh_win_size_min;
  p.adaptiveThreshWinSizeMax = c.adaptive_thresh_win_size_max;
  p.adaptiveThreshWinSizeStep = c.adaptive_thresh_win_size_step;
  // Corner refinement is done by us on the full-resolution image (see refineCorners),
  // so OpenCV's built-in refinement stays at its default (CORNER_REFINE_NONE).
}

void validate(const MarkerDetectorConfig & c)
{
  if (!(c.detection_scale > 0.0 && c.detection_scale <= 1.0)) {
    throw std::invalid_argument("detection_scale must be in (0, 1]");
  }
  if (!(c.min_marker_perimeter_rate > 0.0)) {
    throw std::invalid_argument("min_marker_perimeter_rate must be > 0");
  }
  if (!(c.max_marker_perimeter_rate >= c.min_marker_perimeter_rate)) {
    throw std::invalid_argument("max_marker_perimeter_rate must be >= min_marker_perimeter_rate");
  }
  if (!(c.polygonal_approx_accuracy_rate > 0.0 && c.polygonal_approx_accuracy_rate < 1.0)) {
    throw std::invalid_argument("polygonal_approx_accuracy_rate must be in (0, 1)");
  }
  if (!(c.min_marker_distance_rate >= 0.0 && c.min_marker_distance_rate <= 1.0)) {
    throw std::invalid_argument("min_marker_distance_rate must be in [0, 1]");
  }
  if (c.min_distance_to_border < 0) {
    throw std::invalid_argument("min_distance_to_border must be >= 0");
  }
  if (!(c.min_corner_distance_rate >= 0.0 && c.min_corner_distance_rate <= 1.0)) {
    throw std::invalid_argument("min_corner_distance_rate must be in [0, 1]");
  }
  if (!(c.error_correction_rate >= 0.0 && c.error_correction_rate <= 1.0)) {
    throw std::invalid_argument("error_correction_rate must be in [0, 1]");
  }
  if (!(c.max_erroneous_bits_in_border_rate >= 0.0 && c.max_erroneous_bits_in_border_rate <= 1.0)) {
    throw std::invalid_argument("max_erroneous_bits_in_border_rate must be in [0, 1]");
  }
  if (!(c.min_otsu_std_dev >= 0.0)) {
    throw std::invalid_argument("min_otsu_std_dev must be >= 0");
  }
  if (c.corner_refinement_method < 0 || c.corner_refinement_method > 2) {
    throw std::invalid_argument("corner_refinement_method must be 0, 1 or 2");
  }
  if (c.adaptive_thresh_constant < 0 || c.adaptive_thresh_constant > 255) {
    throw std::invalid_argument("adaptive_thresh_constant must be in [0, 255]");
  }
  if (c.adaptive_thresh_win_size_min < 3 ||
    c.adaptive_thresh_win_size_max < c.adaptive_thresh_win_size_min ||
    c.adaptive_thresh_win_size_step < 1)
  {
    throw std::invalid_argument(
            "need adaptive_thresh_win_size_min >= 3, max >= min and step >= 1");
  }
}

/// cornerSubPix on the full-resolution image. The search window is tied to the size of one
/// marker module in pixels: it has to stay inside the black border (1 module wide), otherwise
/// the inner pattern pulls the corner away.
void refineCorners(const cv::Mat & gray, std::vector<MarkerDetection> & detections, int modules_across)
{
  const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);
  std::vector<cv::Point2f> pts(4);

  for (auto & d : detections) {
    double min_side = 1e30;
    for (int j = 0; j < 4; ++j) {
      min_side = std::min(min_side, cv::norm(d.corners[j] - d.corners[(j + 1) % 4]));
    }
    const double module_px = min_side / modules_across;
    const int half_win = std::min(8, static_cast<int>(std::lround(0.3 * module_px)));
    if (half_win < 2) {
      continue;  // marker too small for a meaningful refinement, keep the coarse corners
    }

    std::copy(d.corners.begin(), d.corners.end(), pts.begin());
    cv::cornerSubPix(gray, pts, cv::Size(half_win, half_win), cv::Size(-1, -1), criteria);
    std::copy(pts.begin(), pts.end(), d.corners.begin());
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------

struct MarkerDetector::Impl
{
  explicit Impl(const MarkerDetectorConfig & c)
  : cfg(c)
  {
    validate(cfg);
    const int dict_id = predefinedDictionaryId(cfg.dictionary);
#ifdef ARUCO_TRACKER_NEW_ARUCO_API
    const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dict_id);
    cv::aruco::DetectorParameters params;
    applyConfig(params, cfg);
    detector = std::make_unique<cv::aruco::ArucoDetector>(dictionary, params);
    modules_across = dictionary.markerSize + 2;  // +2 = the black border
#else
    dictionary = cv::aruco::getPredefinedDictionary(dict_id);
    params = cv::aruco::DetectorParameters::create();
    applyConfig(*params, cfg);
    modules_across = dictionary->markerSize + 2;
#endif
  }

  MarkerDetectorConfig cfg;
  int modules_across{6};

#ifdef ARUCO_TRACKER_NEW_ARUCO_API
  std::unique_ptr<cv::aruco::ArucoDetector> detector;
#else
  cv::Ptr<cv::aruco::Dictionary> dictionary;
  cv::Ptr<cv::aruco::DetectorParameters> params;
#endif

  // Buffers kept between frames to avoid per-frame allocations
  cv::Mat scaled;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> rejected;
};

MarkerDetector::MarkerDetector(const MarkerDetectorConfig & config)
: impl_(std::make_unique<Impl>(config)) {}

MarkerDetector::~MarkerDetector() = default;

std::vector<MarkerDetection> MarkerDetector::detect(const cv::Mat & gray, size_t * rejected_count)
{
  CV_Assert(gray.type() == CV_8UC1);
  Impl & s = *impl_;

  if (rejected_count != nullptr) {
    *rejected_count = 0;
  }

  const double scale = s.cfg.detection_scale;
  const bool downscale = scale < 0.999;

  if (downscale) {
    cv::resize(gray, s.scaled, cv::Size(), scale, scale, cv::INTER_AREA);
  }
  const cv::Mat & search_img = downscale ? s.scaled : gray;

  s.corners.clear();
  s.ids.clear();
  s.rejected.clear();
#ifdef ARUCO_TRACKER_NEW_ARUCO_API
  s.detector->detectMarkers(search_img, s.corners, s.ids, s.rejected);
  if (rejected_count != nullptr) {
    *rejected_count = s.rejected.size();
  }
#else
  cv::aruco::detectMarkers(search_img, s.dictionary, s.corners, s.ids, s.params, s.rejected);
  if (rejected_count != nullptr) {
    *rejected_count = s.rejected.size();
  }
#endif

  std::vector<MarkerDetection> out;
  out.reserve(s.ids.size());

  // cv::resize maps pixel centres: dst = (src + 0.5) * scale - 0.5. Invert exactly, otherwise
  // a downscaled search would be off by (1/scale - 1) / 2 pixels.
  const double inv = 1.0 / scale;
  for (size_t i = 0; i < s.ids.size(); ++i) {
    MarkerDetection d;
    d.id = s.ids[i];
    for (int j = 0; j < 4; ++j) {
      const cv::Point2f p = s.corners[i][j];
      d.corners[j] = downscale ?
        cv::Point2f(
        static_cast<float>((p.x + 0.5) * inv - 0.5),
        static_cast<float>((p.y + 0.5) * inv - 0.5)) :
        p;
    }
    out.push_back(d);
  }

  if (s.cfg.subpixel_refine && !out.empty()) {
    refineCorners(gray, out, s.modules_across);
  }
  return out;
}

void drawDetections(cv::Mat & bgr, const std::vector<MarkerDetection> & detections, double scale)
{
  const cv::Scalar green(0, 255, 0);
  const cv::Scalar red(0, 0, 255);
  const cv::Scalar yellow(0, 255, 255);

  for (const auto & d : detections) {
    cv::Point poly[4];
    cv::Point2f centre(0.f, 0.f);
    for (int j = 0; j < 4; ++j) {
      poly[j] = cv::Point(cvRound(d.corners[j].x * scale), cvRound(d.corners[j].y * scale));
      centre += d.corners[j] * 0.25f;
    }
    for (int j = 0; j < 4; ++j) {
      cv::line(bgr, poly[j], poly[(j + 1) % 4], green, 2, cv::LINE_AA);
    }
    cv::circle(bgr, poly[0], 4, red, cv::FILLED, cv::LINE_AA);
    cv::putText(
      bgr, std::to_string(d.id),
      cv::Point(cvRound(centre.x * scale) - 8, cvRound(centre.y * scale) + 8),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2, cv::LINE_AA);
  }
}

}  // namespace aruco_tracker
