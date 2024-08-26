#include "utils.h"

void generate_point_cloud_3d(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc) {
  // generate point cloud in a square
  const double resolution = 0.1;
  const double side_len = 5;
  uint8_t color[kOptValNum] = {255, 0, 0};
  for (double z = 0.; z < side_len; z += (5 * resolution)) {
    for (double x = 0.; x < side_len; x += resolution) {
      for (double y = 0.; y < side_len; y += resolution) {
        pcl::PointXYZRGB p(color[0], color[1], color[2]);
        p.x = x;
        p.y = y;
        p.z = z;
        pc->push_back(p);
      }
    }
  }
}

void show_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pc,
                      const std::string& title) {
  pcl::visualization::PCLVisualizer viewer(title);
  viewer.setBackgroundColor(0, 0, 0);
  viewer.addPointCloud<pcl::PointXYZRGB>(pc, "", 0);
  while (!viewer.wasStopped()) {
    viewer.spinOnce(1000);
  }
}

void generate_affine_matrix(const Eigen::Quaterniond& q,
                            const Eigen::Vector3d& t, Eigen::Matrix4d* T) {
  (*T) = Eigen::Matrix4d::Identity();
  (*T).block<3, 3>(0, 0) = q.toRotationMatrix();
  (*T).block<3, 1>(0, 3) = t;
}

void generate_affine_matrix(const double q[4],
                            const double t[3], Eigen::Matrix4d* T) {
  Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
  Eigen::Vector3d trans(t[0], t[1], t[2]);
  (*T) = Eigen::Matrix4d::Identity();
  (*T).block<3, 3>(0, 0) = quat.toRotationMatrix();
  (*T).block<3, 1>(0, 3) = trans;
}

void transform_point_cloud(const Eigen::Matrix4d& T,
                           pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr src,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr tar) {
  // if (tar->empty()) {
  //   tar = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(
  //       new pcl::PointCloud<pcl::PointXYZRGB>);
  // }
  pcl::transformPointCloud(*src, *tar, T);
}

void find_point_correspondences(const Eigen::Matrix4d& guess_T,
                                pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr src,
                                pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr tar,
                                std::vector<Corres3d>* corres) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr src2tar(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  transform_point_cloud(guess_T, src, src2tar);

  // brute force search，kd tree is better
  for (int i = 0; i < src2tar->size(); ++i) {
    double min_dis = DBL_MAX;
    int nearest_id = INT_MIN;
    for (int j = 0; j < tar->size(); ++j) {
      double cur_dis = cal_dist((*src2tar)[i], (*tar)[j]);
      if (cur_dis < min_dis) {
        min_dis = cur_dis;
        nearest_id = j;
      }
    }
    Corres3d correspondence((*src2tar)[i].x, (*src2tar)[i].y, (*src2tar)[i].z,
                            (*tar)[nearest_id].x, (*tar)[nearest_id].y, (*tar)[nearest_id].z,
                            min_dis);
    corres->push_back(correspondence);
  }
}

double cal_dist(const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) {
  const Eigen::Vector3f diff(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
  return diff.norm();
}

void set_point_cloud_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc, uint8_t r,
                           uint8_t g, uint8_t b) {
  for (auto& p : *pc) {
    p.r = r;
    p.g = g;
    p.b = b;
  }
}

void generate_quat_trans(const double angle_trans[kOptValNum], Eigen::Quaterniond *q, Eigen::Vector3d *t){
  Eigen::AngleAxisd yaw(angle_trans[0], Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitch(angle_trans[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd roll(angle_trans[2], Eigen::Vector3d::UnitX());
  (*q) = yaw * pitch * roll;
  (*t) << angle_trans[3], angle_trans[4], angle_trans[5];
}
