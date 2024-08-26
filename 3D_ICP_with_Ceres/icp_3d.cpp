#include "icp_3d.h"

#include <iostream>
#include <string>
#include <vector>

#include "utils.h"

constexpr double kRoll = 30.;
constexpr double kPitch = 45.;
constexpr double kYaw = 60.;
constexpr double kX = 10.;
constexpr double kY = 20.;
constexpr double kZ = 30.;
constexpr int    kMaxIteration = 20;
double ceres_solve_icp_3d(const double* init_guess_yaw_t,
                          const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr src,
                          const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr tar,
                          double* final_guess_yaw_t);

int main() {
  // generate two frame point cloud
  // angle to quat: https://blog.csdn.net/DaqianC/article/details/81474338
  double true_angel_trans[kOptValNum] = {kYaw * kDegree2Rad,
                                         kPitch * kDegree2Rad,
                                         kRoll * kDegree2Rad,
                                         kX,
                                         kY,
                                         kZ};
  Eigen::Quaterniond true_quat;
  Eigen::Vector3d true_trans;
  generate_quat_trans(true_angel_trans, &true_quat, &true_trans);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_pc(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  generate_point_cloud_3d(src_pc);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tar_pc(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::transformPointCloud(*src_pc, *tar_pc, true_trans, true_quat);

  // take a look at the initial two point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr merge_pc(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  double init_angle_trans[kOptValNum] = {(kYaw - 5) * kDegree2Rad,
                                         (kPitch - 5) * kDegree2Rad,
                                         (kRoll - 5.) * kDegree2Rad,
                                         kX - 3,
                                         kY - 3,
                                         kZ - 3};
  Eigen::Quaterniond init_quat;
  Eigen::Vector3d init_trans;
  generate_quat_trans(init_angle_trans, &init_quat, &init_trans);
  pcl::transformPointCloud(*src_pc, *merge_pc, init_trans, init_quat);
  set_point_cloud_color(merge_pc, 0, 255, 0);
  *merge_pc += (*tar_pc);
  std::string title = "init point cloud";
  show_point_cloud(merge_pc, title);

  // Ceres find the transform
  double final_angle_trans[kOptValNum] = {0.};
  ceres_solve_icp_3d(init_angle_trans, src_pc, tar_pc, final_angle_trans);
  cout << "final guess: " ;
  for( int i=0; i<kOptValNum; ++i ){
    if(i<3)
      cout<<final_angle_trans[i]/kDegree2Rad<<" ";
    else
      cout<<final_angle_trans[i]<<" ";
  }
  cout << endl;

  // take a look at the final two point cloud.
  Eigen::Quaterniond final_quat;
  Eigen::Vector3d final_trans;
  generate_quat_trans(final_angle_trans, &final_quat, &final_trans);
  pcl::transformPointCloud(*src_pc, *merge_pc, final_trans, final_quat);
  set_point_cloud_color(merge_pc, 0, 0, 255);
  *merge_pc += (*tar_pc);
  title = "final point cloud";
  show_point_cloud(merge_pc, title);

  return 0;
}

double ceres_solve_icp_3d(const double* init_angle_trans,
                          const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr src,
                          const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr tar,
                          double* final_angle_trans) {
  double each_step_quat[4] = {1., 0., 0., 0.};
  double each_step_trans[3] = {0., 0., 0.};

  // updated transform in each step
  Eigen::Quaterniond init_quat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d init_trans = Eigen::Vector3d::Identity();
  generate_quat_trans(init_angle_trans, &init_quat, &init_trans);
  Eigen::Matrix4d final_guess_T = Eigen::Matrix4d::Identity();
  generate_affine_matrix(init_quat, init_trans, &final_guess_T);

  int max_ite = 0;
  while (max_ite < kMaxIteration) {
    ++max_ite;

    // ICP find correspondences.
    std::vector<Corres3d> corres;
    find_point_correspondences(final_guess_T, src, tar, &corres);

    /******** Ceres Solver *************/
    // build problem
    ceres::Problem problem;
    for (const auto& cor : corres) {
      Eigen::Vector3d src;
      Eigen::Vector3d tar;
      src << cor.x1, cor.y1, cor.z1;
      tar << cor.x2, cor.y2, cor.z2;
// #define AUTO_DIFF
#ifdef AUTO_DIFF
      ceres::CostFunction* cost_function = ICP3DResiaual::Create(src, tar);
      problem.AddResidualBlock(cost_function, nullptr, each_step_quat, each_step_trans);
#else
      ICP3DResiaual* cost_function = new ICP3DResiaual(src, tar);
      problem.AddResidualBlock(cost_function, nullptr, each_step_quat, each_step_trans);
#endif
    }
    // problem.AddParameterBlock(each_step_quat, 4, local_parameterization);
    ceres::LocalParameterization * quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( each_step_quat, quaternion_parameterization );

    // solve problem
    ceres::Solver::Options options;
    // options.minimizer_progress_to_stdout = true; //true;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = 6;
    options.max_num_iterations = 500;
    ceres::Solver::Summary summary;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-20;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.min_line_search_step_size = 1e-3;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;
    /******** Ceres Solver *************/

    // Update Transform
    Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
    generate_affine_matrix(each_step_quat, each_step_trans, &delta_T);
    final_guess_T = delta_T * final_guess_T;

    // if loop can break?
    const Eigen::Vector3d ypr =
        Eigen::Quaterniond(each_step_quat[0], each_step_quat[1],
                           each_step_quat[2], each_step_quat[3])
            .toRotationMatrix()
            .eulerAngles(2, 1, 0);
    const double each_step_diff[kOptValNum] = {ypr(0),
                                     ypr(1),
                                     ypr(2),
                                     each_step_trans[0],
                                     each_step_trans[1],
                                     each_step_trans[2]};
    bool can_break = true;
    for (int i = 0; i < kOptValNum; ++i) {
      if (each_step_diff[i] > KConverge[i]) {
        can_break = false;
      }
    }
    if (can_break) {
      break;
    }

    // inilitize each step delta
    each_step_quat[0] = 1;
    for( int i=0; i<3; ++i ){
      each_step_trans[i] = 0.;
      each_step_quat[i+1] = 0.;
    }

  }
  const Eigen::Vector3d ypr = final_guess_T.block<3,3>(0,0).eulerAngles(2, 1, 0);
  final_angle_trans[0] = ypr[0];
  final_angle_trans[1] = ypr[1];
  final_angle_trans[2] = ypr[2]; // ypr means R(y)*R(p)*R(r)
  final_angle_trans[3] = final_guess_T(0,3);
  final_angle_trans[4] = final_guess_T(1,3);
  final_angle_trans[5] = final_guess_T(2,3);
}