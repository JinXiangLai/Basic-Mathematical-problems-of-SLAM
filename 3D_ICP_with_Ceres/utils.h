#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

constexpr double kDegree2Rad = M_PI / 180;
constexpr int    kOptValNum = 6;
constexpr double KConverge[kOptValNum] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5};

struct Corres3d {
  double x1;
  double y1;
  double z1;
  double x2;
  double y2;
  double z2;
  double dist;
  Corres3d() = delete;
  Corres3d(double sx, double sy, double sz, double tx, double ty, double tz, double dis = 0.)
      : x1(sx), y1(sy), z1(sz), x2(tx), y2(ty), z2(tz), dist(dis) {}
};

void generate_point_cloud_3d(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc);

void show_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pc,
                      const std::string& title);

void generate_affine_matrix(const Eigen::Quaterniond& q,
                            const Eigen::Vector3d& t, Eigen::Matrix4d* T);

void generate_affine_matrix(const double q[4],
                            const double t[3], Eigen::Matrix4d* T);

void transform_point_cloud(const Eigen::Matrix4d& T,
                           pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr src,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr tar);

void find_point_correspondences(const Eigen::Matrix4d& guess_T,
                                pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr src,
                                pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr tar,
                                std::vector<Corres3d>* corres);

double cal_dist(const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2);

void set_point_cloud_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc, uint8_t r,
                           uint8_t g, uint8_t b);

void generate_quat_trans(const double[kOptValNum], Eigen::Quaterniond* q, Eigen::Vector3d* t);