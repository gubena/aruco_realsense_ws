#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "geometry_msgs/msg/pose_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"

namespace
{
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

struct PlaneReference
{
  int id{0};
  Vector2d plane_point{0.0, 0.0};
};

struct PlaneBasis
{
  Vector3d origin{0.0, 0.0, 0.0};
  Vector3d x_axis{1.0, 0.0, 0.0};
  Vector3d y_axis{0.0, 1.0, 0.0};
};

struct Affine2D
{
  Matrix2d A = Matrix2d::Identity();
  Vector2d b{0.0, 0.0};
};

Vector3d posePosition(const geometry_msgs::msg::Pose & p)
{
  return Vector3d(p.position.x, p.position.y, p.position.z);
}

std::map<int, PlaneReference> loadReferenceMarkers(const YAML::Node & cfg)
{
  std::map<int, PlaneReference> refs;
  const YAML::Node items = cfg["reference_markers"];
  if (!items || !items.IsSequence()) {
    return refs;
  }

  for (const auto & item : items) {
    PlaneReference ref;
    ref.id = item["id"].as<int>();

    std::vector<double> xy;
    if (item["position"]) {
      xy = item["position"].as<std::vector<double>>();
    } else if (item["plane_position"]) {
      xy = item["plane_position"].as<std::vector<double>>();
    } else {
      throw std::runtime_error("Each reference marker must define 'position' or 'plane_position'");
    }

    if (xy.size() != 2) {
      throw std::runtime_error("Reference marker position must have exactly 2 values [x, y]");
    }

    ref.plane_point = Vector2d(xy[0], xy[1]);
    refs[ref.id] = ref;
  }
  return refs;
}

std::map<int, double> loadElevatedMarkers(const YAML::Node & cfg)
{
  std::map<int, double> elevated;
  const YAML::Node items = cfg["elevated_markers"];
  if (!items || !items.IsSequence()) {
    return elevated;
  }

  for (const auto & item : items) {
    const int id = item["id"].as<int>();
    const double height = item["height"] ? item["height"].as<double>() : 0.0;
    elevated[id] = height;
  }
  return elevated;
}

PlaneBasis estimatePlaneBasisFromPoints(const std::vector<Vector3d> & points)
{
  if (points.size() < 3) {
    throw std::runtime_error("At least 3 3D points are required to estimate a plane basis");
  }

  Vector3d centroid = Vector3d::Zero();
  for (const auto & p : points) {
    centroid += p;
  }
  centroid /= static_cast<double>(points.size());

  Vector3d v0 = points[1] - points[0];
  if (v0.norm() < 1e-8) {
    v0 = points[2] - points[0];
  }
  if (v0.norm() < 1e-8) {
    throw std::runtime_error("Detected marker points are degenerate");
  }

  Vector3d x_axis = v0.normalized();
  Vector3d normal = (points[2] - points[0]).cross(x_axis);
  if (normal.norm() < 1e-8) {
    throw std::runtime_error("Detected marker points are collinear");
  }
  normal.normalize();
  Vector3d y_axis = normal.cross(x_axis).normalized();

  PlaneBasis basis;
  basis.origin = centroid;
  basis.x_axis = x_axis;
  basis.y_axis = y_axis;
  return basis;
}

Vector2d projectToPlane2D(const Vector3d & p, const PlaneBasis & basis)
{
  const Vector3d d = p - basis.origin;
  return Vector2d(d.dot(basis.x_axis), d.dot(basis.y_axis));
}

Affine2D solveAffine2D(
  const std::vector<Vector2d> & src,
  const std::vector<Vector2d> & dst)
{
  if (src.size() != dst.size() || src.size() < 3) {
    throw std::runtime_error("Need at least 3 corresponding points to estimate an affine planar transform");
  }

  Eigen::MatrixXd A(2 * static_cast<int>(src.size()), 6);
  Eigen::VectorXd b(2 * static_cast<int>(src.size()));

  for (size_t i = 0; i < src.size(); ++i) {
    const double sx = src[i].x();
    const double sy = src[i].y();
    const double dx = dst[i].x();
    const double dy = dst[i].y();

    A(2 * i + 0, 0) = sx;
    A(2 * i + 0, 1) = sy;
    A(2 * i + 0, 2) = 1.0;
    A(2 * i + 0, 3) = 0.0;
    A(2 * i + 0, 4) = 0.0;
    A(2 * i + 0, 5) = 0.0;

    A(2 * i + 1, 0) = 0.0;
    A(2 * i + 1, 1) = 0.0;
    A(2 * i + 1, 2) = 0.0;
    A(2 * i + 1, 3) = sx;
    A(2 * i + 1, 4) = sy;
    A(2 * i + 1, 5) = 1.0;

    b(2 * i + 0) = dx;
    b(2 * i + 1) = dy;
  }

  Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  Affine2D affine;
  affine.A << x[0], x[1], x[3], x[4];
  affine.b << x[2], x[5];
  return affine;
}

Vector2d applyAffine2D(const Vector2d & p, const Affine2D & affine)
{
  return affine.A * p + affine.b;
}

geometry_msgs::msg::Pose makePoseFromPlanePoint(const Vector2d & p, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = p.x();
  pose.position.y = p.y();
  pose.position.z = z;
  pose.orientation.w = 1.0;
  pose.orientation.x = 0.0;
  pose.orientation.y = 0.0;
  pose.orientation.z = 0.0;
  return pose;
}

geometry_msgs::msg::Pose makePoseFromPlanePoint(const Vector2d & p)
{
  return makePoseFromPlanePoint(p, 0.0);
}

}  // namespace

class PlaneTransformNode : public rclcpp::Node
{
public:
  PlaneTransformNode()
  : Node("plane_transform_node")
  {
    declare_parameter<std::string>("config_file", "");
    const std::string cfg_path = get_parameter("config_file").as_string();
    if (cfg_path.empty()) {
      throw std::runtime_error("Missing required parameter: config_file");
    }

    const YAML::Node cfg = YAML::LoadFile(cfg_path);
    reference_markers_ = loadReferenceMarkers(cfg);
    if (reference_markers_.empty()) {
      throw std::runtime_error("No reference markers were found in the config file");
    }

    plane_frame_ = cfg["plane_frame"] ? cfg["plane_frame"].as<std::string>() : "aruco_plane";

    const std::string poses_topic = cfg["input"]["poses_topic"] ? cfg["input"]["poses_topic"].as<std::string>() : "/aruco_detector/poses";
    const std::string ids_topic = cfg["input"]["ids_topic"] ? cfg["input"]["ids_topic"].as<std::string>() : "/aruco_detector/pose_ids";
    const std::string output_topic = cfg["output"]["transformed_poses_topic"] ? cfg["output"]["transformed_poses_topic"].as<std::string>() : "/aruco_plane_localizer/transformed_poses";
    const std::string elevated_topic = cfg["output"]["elevated_poses_topic"] ? cfg["output"]["elevated_poses_topic"].as<std::string>() : "/aruco_plane_localizer/elevated_poses";

    poses_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      poses_topic,
      rclcpp::SensorDataQoS(),
      std::bind(&PlaneTransformNode::onPoses, this, std::placeholders::_1));
    ids_sub_ = create_subscription<std_msgs::msg::Int32MultiArray>(
      ids_topic,
      rclcpp::SensorDataQoS(),
      std::bind(&PlaneTransformNode::onIds, this, std::placeholders::_1));
    transformed_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(output_topic, 10);
    elevated_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(elevated_topic, 10);

    elevated_markers_ = loadElevatedMarkers(cfg);

    RCLCPP_INFO(get_logger(), "Loaded %zu reference markers and %zu elevated markers for plane localization", reference_markers_.size(), elevated_markers_.size());
  }

private:
  void onPoses(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    last_poses_ = *msg;
    have_poses_ = true;
    updateIfReady();
  }

  void onIds(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
  {
    last_ids_ = *msg;
    have_ids_ = true;
    updateIfReady();
  }

  void updateIfReady()
  {
    if (!have_poses_ || !have_ids_) {
      return;
    }
    if (last_poses_.poses.size() != last_ids_.data.size()) {
      return;
    }

    std::map<int, Vector3d> detected_points;
    for (size_t i = 0; i < last_poses_.poses.size(); ++i) {
      const int id = last_ids_.data[i];
      detected_points[id] = posePosition(last_poses_.poses[i]);
    }

    std::vector<int> visible_ref_ids;
    for (const auto & [id, ref] : reference_markers_) {
      (void)ref;
      if (detected_points.count(id) > 0) {
        visible_ref_ids.push_back(id);
      }
    }
    std::sort(visible_ref_ids.begin(), visible_ref_ids.end());

    if (visible_ref_ids.size() != 4) {
      return;
    }

    std::vector<Vector3d> source_points;
    std::vector<Vector2d> source_plane_points;
    std::vector<Vector2d> target_plane_points;
    source_points.reserve(visible_ref_ids.size());
    source_plane_points.reserve(visible_ref_ids.size());
    target_plane_points.reserve(visible_ref_ids.size());

    for (const int id : visible_ref_ids) {
      source_points.push_back(detected_points.at(id));
    }

    const PlaneBasis basis = estimatePlaneBasisFromPoints(source_points);
    for (const int id : visible_ref_ids) {
      source_plane_points.push_back(projectToPlane2D(detected_points.at(id), basis));
      target_plane_points.push_back(reference_markers_.at(id).plane_point);
    }

    try {
      const Affine2D affine = solveAffine2D(source_plane_points, target_plane_points);

      geometry_msgs::msg::PoseArray out;
      out.header.stamp = last_poses_.header.stamp;
      out.header.frame_id = plane_frame_;

      for (const int id : visible_ref_ids) {
        const Vector2d local = projectToPlane2D(detected_points.at(id), basis);
        const Vector2d mapped = applyAffine2D(local, affine);
        out.poses.push_back(makePoseFromPlanePoint(mapped));
      }

      transformed_pub_->publish(out);

      if (!elevated_markers_.empty()) {
        geometry_msgs::msg::PoseArray elevated_out;
        elevated_out.header.stamp = last_poses_.header.stamp;
        elevated_out.header.frame_id = plane_frame_;

        for (const auto & [id, height] : elevated_markers_) {
          if (detected_points.count(id) == 0) {
            continue;
          }
          const Vector2d local = projectToPlane2D(detected_points.at(id), basis);
          const Vector2d mapped = applyAffine2D(local, affine);
          elevated_out.poses.push_back(makePoseFromPlanePoint(mapped, height));
        }
        elevated_pub_->publish(elevated_out);
      }

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "Published corrected planar positions for %zu markers", out.poses.size());
    } catch (const std::exception & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Failed to fit the plane transform: %s", ex.what());
    }
  }

  std::map<int, PlaneReference> reference_markers_;
  std::map<int, double> elevated_markers_;
  std::string plane_frame_ = "aruco_plane";

  geometry_msgs::msg::PoseArray last_poses_;
  std_msgs::msg::Int32MultiArray last_ids_;
  bool have_poses_{false};
  bool have_ids_{false};

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr poses_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr ids_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr transformed_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr elevated_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PlaneTransformNode>());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("plane_transform_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
