// Step 1: subscribe to a raw image stream, detect ArUco 4x4 markers, report them.
//
//   subscribes : <image_topic>            sensor_msgs/Image   (best-effort, depth 1)
//   publishes  : ~/debug_image            sensor_msgs/Image   (bgr8, only while subscribed)
//
// The image is wrapped zero-copy (no cv_bridge), converted to gray and handed to MarkerDetector.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

#include "aruco_tracker/camera_intrinsics.hpp"
#include "aruco_tracker/marker_detector.hpp"
#include "aruco_tracker/pose_estimator.hpp"

namespace aruco_tracker
{

class ArucoDetectorNode : public rclcpp::Node
{
  using Image = sensor_msgs::msg::Image;
  using Clock = std::chrono::steady_clock;

public:
  ArucoDetectorNode()
  : rclcpp::Node("aruco_detector")
  {
    // ---- parameters -------------------------------------------------------------------------
    const auto image_topic = declare_parameter<std::string>("image_topic", "image_raw");
    const auto calib_file = declare_parameter<std::string>("camera_info_file", "");
    publish_debug_ = declare_parameter<bool>("publish_debug_image", true);
    debug_scale_ = std::clamp(declare_parameter<double>("debug_image_scale", 0.5), 0.05, 1.0);
    debug_logging_ = declare_parameter<bool>("debug_logging", false);

    MarkerDetectorConfig cfg;
    cfg.dictionary = declare_parameter<std::string>("dictionary", cfg.dictionary);
    cfg.detection_scale = declare_parameter<double>("detection_scale", cfg.detection_scale);
    cfg.subpixel_refine = declare_parameter<bool>("subpixel_refine", cfg.subpixel_refine);
    cfg.min_marker_perimeter_rate = declare_parameter<double>(
      "min_marker_perimeter_rate", cfg.min_marker_perimeter_rate);
    cfg.max_marker_perimeter_rate = declare_parameter<double>(
      "max_marker_perimeter_rate", cfg.max_marker_perimeter_rate);
    cfg.polygonal_approx_accuracy_rate = declare_parameter<double>(
      "polygonal_approx_accuracy_rate", cfg.polygonal_approx_accuracy_rate);
    cfg.min_marker_distance_rate = declare_parameter<double>(
      "min_marker_distance_rate", cfg.min_marker_distance_rate);
    cfg.min_distance_to_border = declare_parameter<int>(
      "min_distance_to_border", cfg.min_distance_to_border);
    cfg.min_corner_distance_rate = declare_parameter<double>(
      "min_corner_distance_rate", cfg.min_corner_distance_rate);
    cfg.error_correction_rate = declare_parameter<double>(
      "error_correction_rate", cfg.error_correction_rate);
    cfg.max_erroneous_bits_in_border_rate = declare_parameter<double>(
      "max_erroneous_bits_in_border_rate", cfg.max_erroneous_bits_in_border_rate);
    cfg.min_otsu_std_dev = declare_parameter<double>(
      "min_otsu_std_dev", cfg.min_otsu_std_dev);
    cfg.corner_refinement_method = declare_parameter<int>(
      "corner_refinement_method", cfg.corner_refinement_method);
    cfg.adaptive_thresh_constant = declare_parameter<int>(
      "adaptive_thresh_constant", cfg.adaptive_thresh_constant);
    cfg.adaptive_thresh_win_size_min = declare_parameter<int>(
      "adaptive_thresh_win_size_min", cfg.adaptive_thresh_win_size_min);
    cfg.adaptive_thresh_win_size_max = declare_parameter<int>(
      "adaptive_thresh_win_size_max", cfg.adaptive_thresh_win_size_max);
    cfg.adaptive_thresh_win_size_step = declare_parameter<int>(
      "adaptive_thresh_win_size_step", cfg.adaptive_thresh_win_size_step);
    marker_size_m_ = declare_parameter<double>("marker_size", 0.05);
    publish_poses_ = declare_parameter<bool>("publish_poses", true);
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    tf_child_prefix_ = declare_parameter<std::string>("tf_child_prefix", "aruco_");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "");
    max_reprojection_error_px_ = declare_parameter<double>("max_reprojection_error_px", 3.0);

    detector_ = std::make_unique<MarkerDetector>(cfg);  // throws on a bad config

    RCLCPP_INFO(
      get_logger(),
      "Detector params: detection_scale=%.2f, perim=[%.3f, %.3f], approx=%.3f, min_dist=%.3f, border=%d, corner_refine=%d, adaptive_thresh=(%d,%d,%d,%d), error_corr=%.2f, border_err=%.2f, otsu_std=%.1f",
      cfg.detection_scale, cfg.min_marker_perimeter_rate, cfg.max_marker_perimeter_rate,
      cfg.polygonal_approx_accuracy_rate, cfg.min_marker_distance_rate,
      cfg.min_distance_to_border, cfg.corner_refinement_method,
      cfg.adaptive_thresh_constant, cfg.adaptive_thresh_win_size_min,
      cfg.adaptive_thresh_win_size_max, cfg.adaptive_thresh_win_size_step,
      cfg.error_correction_rate, cfg.max_erroneous_bits_in_border_rate,
      cfg.min_otsu_std_dev);

    // ---- intrinsics (not needed for detection itself, loaded now for the pose steps) --------
    if (!calib_file.empty()) {
      intrinsics_ = loadCameraIntrinsics(calib_file);  // throws with a readable message
      const auto & K = intrinsics_->camera_matrix;
      RCLCPP_INFO(
        get_logger(), "Intrinsics '%s': %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  model=%s (%d coeffs)",
        intrinsics_->camera_name.c_str(), intrinsics_->image_size.width,
        intrinsics_->image_size.height, K.at<double>(0, 0), K.at<double>(1, 1),
        K.at<double>(0, 2), K.at<double>(1, 2), intrinsics_->distortion_model.c_str(),
        intrinsics_->dist_coeffs.cols);
      if (intrinsics_->camera_matrix.rows == 3 && intrinsics_->camera_matrix.cols == 3) {
        pose_estimator_ = std::make_unique<PoseEstimator>(
          intrinsics_->camera_matrix, intrinsics_->dist_coeffs, marker_size_m_,
          max_reprojection_error_px_);
      }
    } else {
      RCLCPP_INFO(get_logger(), "No camera_info_file given: detection only, no intrinsics loaded.");
      RCLCPP_INFO(get_logger(), "Pose estimation is disabled because no camera intrinsics were loaded.");
    }

    // ---- I/O --------------------------------------------------------------------------------
    // Best-effort + depth 1: if detection is ever slower than the camera we drop frames
    // instead of building up a queue (= growing latency).
    sub_ = create_subscription<Image>(
      image_topic, rclcpp::SensorDataQoS().keep_last(1),
      [this](Image::ConstSharedPtr msg) {onImage(*msg);});

    poses_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("~/poses", 10);
    pose_ids_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("~/pose_ids", 10);

    if (publish_debug_) {
      debug_pub_ = create_publisher<Image>(
        "~/debug_image", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }

    last_report_ = Clock::now();
    RCLCPP_INFO(
      get_logger(), "Listening on '%s' for %s markers (detection_scale=%.2f, subpixel=%s)",
      sub_->get_topic_name(), cfg.dictionary.c_str(), cfg.detection_scale,
      cfg.subpixel_refine ? "on" : "off");
  }

private:
  void onImage(const Image & msg)
  {
    cv::Mat gray;
    if (!toGray(msg, gray)) {
      return;
    }
    checkResolutionOnce(msg);

    const auto t0 = Clock::now();
    size_t rejected_count = 0;
    const std::vector<MarkerDetection> detections = detector_->detect(gray, &rejected_count);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    std::vector<MarkerPose> poses;
    const auto pose_t0 = Clock::now();
    if (pose_estimator_ && pose_estimation_enabled_) {
      poses = pose_estimator_->estimate(detections);
    }
    const double pose_ms = std::chrono::duration<double, std::milli>(Clock::now() - pose_t0).count();

    if (debug_pub_ && debug_pub_->get_subscription_count() > 0) {
      publishDebug(msg, gray, detections, poses);
    }

    if (publish_poses_ && poses_pub_->get_subscription_count() > 0) {
      publishPoses(msg, poses);
    }

    if (publish_tf_ && pose_estimator_ && pose_estimation_enabled_) {
      publishTf(msg, poses);
    }

    // ---- statistics, logged every 2 s --------------------------------------------------------
    ++frames_;
    ms_sum_ += ms;
    ms_max_ = std::max(ms_max_, ms);
    pose_ms_sum_ += pose_ms;
    pose_ms_max_ = std::max(pose_ms_max_, pose_ms);
    const double elapsed = std::chrono::duration<double>(Clock::now() - last_report_).count();
    if (elapsed >= 2.0) {
      if (debug_logging_ || (elapsed >= 2.0 && frames_ == 0)) {
        std::string ids;
        for (const auto & d : detections) {
          ids += (ids.empty() ? "" : ",") + std::to_string(d.id);
        }
        RCLCPP_INFO(
          get_logger(), "Detected: %zu | Rejected: %zu | fps=%.1f | avg=%.1f ms | max=%.1f ms | ids=[%s] | pose_avg=%.1f ms | pose_max=%.1f ms",
          detections.size(), rejected_count, frames_ / elapsed, ms_sum_ / frames_, ms_max_, ids.c_str(),
          pose_ms_sum_ / frames_, pose_ms_max_);
        if (ms_sum_ / frames_ > 30.0) {
          RCLCPP_WARN(
            get_logger(),
            "Detection is too slow for 30 fps. Lower detection_scale (e.g. 0.5) and/or set adaptive_thresh_win_size_min = max = 23.");
        }
      }
      frames_ = 0;
      ms_sum_ = 0.0;
      ms_max_ = 0.0;
      pose_ms_sum_ = 0.0;
      pose_ms_max_ = 0.0;
      last_report_ = Clock::now();
    }
  }

  /// Zero-copy wrap of the message buffer + conversion to 8-bit gray.
  /// On success `gray` is valid as long as `msg` and this node's conv_buf_ are untouched.
  bool toGray(const Image & msg, cv::Mat & gray)
  {
    namespace enc = sensor_msgs::image_encodings;

    int cv_type = 0;
    int cvt_code = -1;  // -1 = already gray
    if (msg.encoding == enc::MONO8) {
      cv_type = CV_8UC1;
    } else if (msg.encoding == enc::BGR8) {
      cv_type = CV_8UC3; cvt_code = cv::COLOR_BGR2GRAY;
    } else if (msg.encoding == enc::RGB8) {
      cv_type = CV_8UC3; cvt_code = cv::COLOR_RGB2GRAY;
    } else if (msg.encoding == enc::BGRA8) {
      cv_type = CV_8UC4; cvt_code = cv::COLOR_BGRA2GRAY;
    } else if (msg.encoding == enc::RGBA8) {
      cv_type = CV_8UC4; cvt_code = cv::COLOR_RGBA2GRAY;
    } else if (msg.encoding == "yuv422_yuy2") {  // YUYV (most USB cameras)
      cv_type = CV_8UC2; cvt_code = cv::COLOR_YUV2GRAY_YUY2;
    } else if (msg.encoding == "yuv422") {        // UYVY
      cv_type = CV_8UC2; cvt_code = cv::COLOR_YUV2GRAY_UYVY;
    } else {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Unsupported image encoding '%s' (supported: mono8, bgr8, rgb8, bgra8, rgba8, "
        "yuv422_yuy2, yuv422)", msg.encoding.c_str());
      return false;
    }

    if (msg.height == 0 || msg.width == 0 ||
      msg.data.size() < static_cast<size_t>(msg.step) * msg.height)
    {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Malformed image message, dropped");
      return false;
    }

    const cv::Mat view(
      static_cast<int>(msg.height), static_cast<int>(msg.width), cv_type,
      const_cast<uint8_t *>(msg.data.data()), msg.step);

    if (cvt_code < 0) {
      gray = view;
    } else {
      cv::cvtColor(view, conv_buf_, cvt_code);
      gray = conv_buf_;
    }
    return true;
  }

  void checkResolutionOnce(const Image & msg)
  {
    if (resolution_checked_) {
      return;
    }
    resolution_checked_ = true;
    RCLCPP_INFO(
      get_logger(), "First frame: %ux%u, encoding '%s'", msg.width, msg.height,
      msg.encoding.c_str());
    if (intrinsics_ && intrinsics_->image_size.area() > 0 &&
      (intrinsics_->image_size.width != static_cast<int>(msg.width) ||
      intrinsics_->image_size.height != static_cast<int>(msg.height)))
    {
      RCLCPP_ERROR(
        get_logger(),
        "Image is %ux%u but the intrinsics were calibrated at %dx%d. They are only valid at the calibration resolution - pose estimation would be wrong.",
        msg.width, msg.height, intrinsics_->image_size.width, intrinsics_->image_size.height);
      pose_estimation_enabled_ = false;
    }
    if (intrinsics_ && intrinsics_->image_size.area() > 0 &&
      (intrinsics_->image_size.width != static_cast<int>(msg.width) ||
      intrinsics_->image_size.height != static_cast<int>(msg.height)))
    {
      RCLCPP_WARN(
        get_logger(),
        "Image is %ux%u but the intrinsics were calibrated at %dx%d. They are only valid at the calibration resolution - pose estimation would be wrong.",
        msg.width, msg.height, intrinsics_->image_size.width, intrinsics_->image_size.height);
    }
  }

  void publishDebug(
    const Image & src, const cv::Mat & gray, const std::vector<MarkerDetection> & detections,
    const std::vector<MarkerPose> & poses)
  {
    const int w = std::max(1, static_cast<int>(std::lround(gray.cols * debug_scale_)));
    const int h = std::max(1, static_cast<int>(std::lround(gray.rows * debug_scale_)));

    auto out = std::make_unique<Image>();
    out->header = src.header;
    out->encoding = sensor_msgs::image_encodings::BGR8;
    out->is_bigendian = false;
    out->width = static_cast<uint32_t>(w);
    out->height = static_cast<uint32_t>(h);
    out->step = static_cast<uint32_t>(w * 3);
    out->data.resize(static_cast<size_t>(out->step) * h);

    cv::Mat bgr(h, w, CV_8UC3, out->data.data());  // writes straight into the message
    if (w == gray.cols && h == gray.rows) {
      cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    } else {
      cv::resize(gray, small_buf_, cv::Size(w, h), 0, 0, cv::INTER_AREA);
      cv::cvtColor(small_buf_, bgr, cv::COLOR_GRAY2BGR);
    }
    if (detections.empty()) {
      const cv::Scalar red(0, 0, 255);
      const int border = std::max(4, std::min(w, h) / 40);
      cv::rectangle(bgr, cv::Point(0, 0), cv::Point(w - 1, h - 1), red, border, cv::LINE_8);
      cv::putText(
        bgr, "NO ARUCO",
        cv::Point(w / 2 - 60, h / 2),
        cv::FONT_HERSHEY_SIMPLEX, 1.0, red, 2, cv::LINE_AA);
    } else {
      drawDetections(bgr, detections, static_cast<double>(w) / gray.cols);
    }

    if (!poses.empty() && intrinsics_ && pose_estimator_) {
      const double scale = static_cast<double>(w) / gray.cols;
      cv::Mat scaled_K = intrinsics_->camera_matrix.clone();
      scaled_K.at<double>(0, 0) *= scale;
      scaled_K.at<double>(1, 1) *= scale;
      scaled_K.at<double>(0, 2) *= scale;
      scaled_K.at<double>(1, 2) *= scale;
      const double axis_length = marker_size_m_ * 0.5;
      for (const auto & pose : poses) {
        cv::Mat rmat;
        cv::Rodrigues(pose.rvec, rmat);
        std::vector<cv::Point3f> axis_points = {
          cv::Point3f(0.0f, 0.0f, 0.0f),
          cv::Point3f(axis_length, 0.0f, 0.0f),
          cv::Point3f(0.0f, axis_length, 0.0f),
          cv::Point3f(0.0f, 0.0f, axis_length),
        };
        std::vector<cv::Point2f> projected_axis;
        cv::projectPoints(
          axis_points, pose.rvec, pose.tvec, scaled_K, intrinsics_->dist_coeffs,
          projected_axis);
        cv::line(
          bgr, cv::Point2f(projected_axis[0].x * scale, projected_axis[0].y * scale),
          cv::Point2f(projected_axis[1].x * scale, projected_axis[1].y * scale),
          cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(
          bgr, cv::Point2f(projected_axis[0].x * scale, projected_axis[0].y * scale),
          cv::Point2f(projected_axis[2].x * scale, projected_axis[2].y * scale),
          cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(
          bgr, cv::Point2f(projected_axis[0].x * scale, projected_axis[0].y * scale),
          cv::Point2f(projected_axis[3].x * scale, projected_axis[3].y * scale),
          cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
      }
    }

    debug_pub_->publish(std::move(out));
  }

  void publishPoses(const Image & msg, const std::vector<MarkerPose> & poses)
  {
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = msg.header.stamp;
    pose_array.header.frame_id = camera_frame_.empty() ? msg.header.frame_id : camera_frame_;
    pose_array.poses.resize(poses.size());

    std_msgs::msg::Int32MultiArray pose_ids;
    pose_ids.data.resize(poses.size());

    for (size_t i = 0; i < poses.size(); ++i) {
      const auto & pose = poses[i];
      pose_ids.data[i] = pose.id;

      tf2::Quaternion q;
      tf2::Matrix3x3 rot;
      cv::Mat rot_matrix;
      cv::Rodrigues(pose.rvec, rot_matrix);
      rot.setValue(
        rot_matrix.at<double>(0, 0), rot_matrix.at<double>(0, 1), rot_matrix.at<double>(0, 2),
        rot_matrix.at<double>(1, 0), rot_matrix.at<double>(1, 1), rot_matrix.at<double>(1, 2),
        rot_matrix.at<double>(2, 0), rot_matrix.at<double>(2, 1), rot_matrix.at<double>(2, 2));
      rot.getRotation(q);

      pose_array.poses[i].position.x = pose.tvec[0];
      pose_array.poses[i].position.y = pose.tvec[1];
      pose_array.poses[i].position.z = pose.tvec[2];
      pose_array.poses[i].orientation.x = q.x();
      pose_array.poses[i].orientation.y = q.y();
      pose_array.poses[i].orientation.z = q.z();
      pose_array.poses[i].orientation.w = q.w();
    }

    poses_pub_->publish(pose_array);
    pose_ids_pub_->publish(pose_ids);
  }

  void publishTf(const Image & msg, const std::vector<MarkerPose> & poses)
  {
    tf2_ros::TransformBroadcaster tf_broadcaster(*this);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = msg.header.stamp;
    transform.header.frame_id = camera_frame_.empty() ? msg.header.frame_id : camera_frame_;

    for (const auto & pose : poses) {
      cv::Mat rot_matrix;
      cv::Rodrigues(pose.rvec, rot_matrix);
      tf2::Matrix3x3 rot(
        rot_matrix.at<double>(0, 0), rot_matrix.at<double>(0, 1), rot_matrix.at<double>(0, 2),
        rot_matrix.at<double>(1, 0), rot_matrix.at<double>(1, 1), rot_matrix.at<double>(1, 2),
        rot_matrix.at<double>(2, 0), rot_matrix.at<double>(2, 1), rot_matrix.at<double>(2, 2));
      tf2::Quaternion q;
      rot.getRotation(q);

      transform.child_frame_id = tf_child_prefix_ + std::to_string(pose.id);
      transform.transform.translation.x = pose.tvec[0];
      transform.transform.translation.y = pose.tvec[1];
      transform.transform.translation.z = pose.tvec[2];
      transform.transform.rotation.x = q.x();
      transform.transform.rotation.y = q.y();
      transform.transform.rotation.z = q.z();
      transform.transform.rotation.w = q.w();
      tf_broadcaster.sendTransform(transform);
    }
  }

  // parameters / members
  bool publish_debug_{true};
  double debug_scale_{0.5};
  bool debug_logging_{false};
  std::unique_ptr<MarkerDetector> detector_;
  std::unique_ptr<PoseEstimator> pose_estimator_;
  std::optional<CameraIntrinsics> intrinsics_;

  rclcpp::Subscription<Image>::SharedPtr sub_;
  rclcpp::Publisher<Image>::SharedPtr debug_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr pose_ids_pub_;

  cv::Mat conv_buf_, small_buf_;
  bool resolution_checked_{false};
  bool pose_estimation_enabled_{true};

  // stats
  Clock::time_point last_report_;
  int frames_{0};
  double ms_sum_{0.0};
  double ms_max_{0.0};
  double pose_ms_sum_{0.0};
  double pose_ms_max_{0.0};

  double marker_size_m_{0.05};
  bool publish_poses_{true};
  bool publish_tf_{true};
  std::string tf_child_prefix_{"aruco_"};
  std::string camera_frame_{};
  double max_reprojection_error_px_{3.0};
};

}  // namespace aruco_tracker

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int ret = 0;
  try {
    rclcpp::spin(std::make_shared<aruco_tracker::ArucoDetectorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("aruco_detector"), "%s", e.what());
    ret = 1;
  }
  rclcpp::shutdown();
  return ret;
}
