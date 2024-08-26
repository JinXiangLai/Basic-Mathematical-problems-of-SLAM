
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <Eigen/Core>

class ICP3DResiaual : public ceres::SizedCostFunction<3, 4, 3> {
 public:
  ICP3DResiaual(const Eigen::Vector3d src_p, const Eigen::Vector3d tar_p)
      : src_p_(src_p), tar_p_(tar_p) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    // initialize parameters
    double const *quat = parameters[0];
    double const *trans = parameters[1];
    const Eigen::Quaterniond q =
        Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]).normalized();
    const Eigen::Vector3d t(trans[0], trans[1], trans[2]);

    // calculate residuals
    const Eigen::Vector3d tar = q * src_p_ + t;
    residuals[0] = tar(0) - tar_p_(0);
    residuals[1] = tar(1) - tar_p_(1);
    residuals[2] = tar(2) - tar_p_(2);

    if (!jacobians) return true;

    const double sx = src_p_(0);
    const double sy = src_p_(1);
    const double sz = src_p_(2);
    const double qw = quat[0];
    const double qx = quat[1];
    const double qy = quat[2];
    const double qz = quat[3];
    const double x = trans[0];
    const double y = trans[1];
    const double z = trans[2];

    // Analytic Jacobians with Multiple Residuals, see:
    // https://groups.google.com/g/ceres-solver/c/nVZdc4hu5zw
    // calculate jacobians
    //     qx  qy  qz  x  y  z (qw = sqrt(1-qx^2-qy^2-qz^2)
    // r1                           ri = tar(i) - tar_p_(i)
    // r2
    // r3
    // dr1/dq
    jacobians[0][0] = 2 * (-qz * sy + qy * sz);  // dr1/dqw
    jacobians[0][1] = 2 * (qy * sy + qz * sz);
    jacobians[0][2] = 2 * (-2 * qy * sx + qx * sy + qw * sz);
    jacobians[0][3] = 2 * (-2 * qz * sx - qw * sy + qw * sz);
    // dr2/dq
    jacobians[0][4] = 2 * (qz * sx - qx * sz);  // dr2/dqw
    jacobians[0][5] = 2 * (qy * sx - 2 * qx * sy - qw * sz);
    jacobians[0][6] = 2 * (qx * sx + qz * sz);
    jacobians[0][7] = 2 * (qw * sx - 2 * qz * sy + qy * sz);
    // dr3/dq
    jacobians[0][8] = 2 * (-qy * sx + qx * sy);  // dr3/dqw
    jacobians[0][9] = 2 * (qz * sx + qw * sy - 2 * qx * sz);
    jacobians[0][10] = 2 * (-qw * sx + qz * sy - 2 * qy * sz);
    jacobians[0][11] = 2 * (qx * sx + qy * sy);

    // dri/dt
    jacobians[1][0] = 1.;  // dr1/dx
    jacobians[1][1] = 0.;
    jacobians[1][2] = 0.;
    jacobians[1][3] = 0.;
    jacobians[1][4] = 1.;  // dr2/dy
    jacobians[1][5] = 0.;
    jacobians[1][6] = 0.;
    jacobians[1][7] = 0.;
    jacobians[1][8] = 1.;  // dr3/dz

    return true;
  }

  template <typename T>
  bool operator()(const T *quat, const T* trans, T *residuals) const {
    Eigen::Matrix<T, 3, 1> t(trans[0], trans[1], trans[2]);
    
    // usage of Eigen::Quaternion: https://www.cnblogs.com/jerry323/p/9097264.html
    // and difference between norm* function: https://blog.csdn.net/m0_56348460/article/details/117386857
    // const Eigen::Quaternion<T> q = Eigen::Quaternion<T>(quat[0], quat[1], quat[2], quat[3]).normalized();
    // const Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
    // const Eigen::Matrix<T, 3, 1> tar = R * src_p_ + t;
    
    // Quat operation in Ceres: https://blog.csdn.net/qq_42731705/article/details/124269886
    Eigen::Matrix<T, 3, 1> tar;
    Eigen::Matrix<T, 3, 1> src(T(src_p_(0)), T(src_p_(1)), T(src_p_(2)));
    ceres::QuaternionRotatePoint(quat, src.data(), tar.data());
    tar += t;
    residuals[0] = T(tar_p_(0)) - tar(0);
    residuals[1] = T(tar_p_(1)) - tar(1);
    residuals[2] = T(tar_p_(2)) - tar(2);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d src_p,
                                     const Eigen::Vector3d tar_p) {
    return (new ceres::AutoDiffCostFunction<ICP3DResiaual, 3, 4, 3>(
        new ICP3DResiaual(src_p, tar_p)));
  }

 private:
  const Eigen::Vector3d src_p_;
  const Eigen::Vector3d tar_p_;
};