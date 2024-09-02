// Created by JinXiangLai
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <vector>

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), twc(t){};
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d twc;
    Pose inverse() const
    {
        return Pose(Rwc.transpose(), -Rwc.transpose() * twc);
    }
    Pose operator*(const Pose& T) const
    {
        return Pose(Rwc * T.Rwc, Rwc * T.twc + twc);
    }
    Eigen::Vector3d operator*(const Eigen::Vector3d& p) const
    {
        return Rwc * p + twc;
    }
};

Eigen::Matrix3d hat(const Eigen::Vector3d a)
{
    Eigen::Matrix3d res;
    res << 0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0;
    return res;
}

Eigen::Vector3d FeatureTriangulationWithEpipolarConstraint(std::vector<Pose> poses, std::vector<Eigen::Vector2d> points)
{
    // Rca * Pa + tca = s * Pc_norm ==>
    // [tca]x * Rca * Pa = s * [tca]x * Pc_norm
    // Pc_norm.T * [tca]x * Rca * Pa = 0
    // 这是对极约束公式，是在已知Xa,Xc的情况下求解Rca
    // 这里，我们已知Xa = [x, y, 1], 所以Pa必须加上这个约束
    // 转换为：
    // Xc.T * [tca]x * Rca * s * Xa = 0
    // 只需要验证 Xc.T * [tca]x * Rca * Xa = 0
    // 则说明不能用这种解法
    const int num = poses.size();
    const Pose Twa = poses[0];
    const Eigen::Vector3d Xa{points[0][0], points[0][1], 1};
    Eigen::MatrixXd A((num - 1), 3);
    A.setZero();
    for (int i = 1; i < num; ++i)
    {
        const Pose Tcw = poses[i].inverse();
        const Pose Tca = Tcw * Twa;
        const Eigen::Matrix3d Rca = Tca.Rwc;
        const Eigen::Vector3d tca = Tca.twc;
        const Eigen::Vector2d& p = points[i];
        const Eigen::Vector3d Pc_norm(p[0], p[1], 1);
        A.block<1, 3>(i - 1, 0) = Pc_norm.transpose() * hat(tca) * Rca;
        // 该式等于0, 因此无法估计出深度sa
        std::cout << "Xc.T * [tca]x * Rca * Xa: " << (Pc_norm.transpose() * hat(tca) * Rca * Xa) << std::endl;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "singular: " << svd.singularValues().transpose() << std::endl;
    const Eigen::Vector3d Pw = Twa * svd.matrixV().col(2);
    return Pw;
}

Eigen::Vector3d FeatureTriangulation2(std::vector<Pose> poses, std::vector<Eigen::Vector2d> points)
{
    // Rca * s2 * Xa + tca = s1 * Pc_norm [1] ==>
    // s2 * [Pc_norm]x * Rca * Xa + [Pc_norm]x * tca = 0
    // s2 = -([Pc_norm]x * tca)[i] / ([Pc_norm]x * Rca * Xa)[i], i=0, 1, 2
    // 这种方法存在的问题是，由于测量噪声的存在，[1]式不一定能精确成立
    // 因此一般假设直接用Pw未知来构建最小二乘解，如下一个三角化函数
    const int num = poses.size();
    const Pose Twa = poses[0];
    const Eigen::Vector3d Xa{points[0][0], points[0][1], 1};
    double s2 = 0;
    for (int i = 1; i < num; ++i)
    {
        const Pose Tcw = poses[i].inverse();
        const Pose Tca = Tcw * Twa;
        const Eigen::Matrix3d Rca = Tca.Rwc;
        const Eigen::Vector3d tca = Tca.twc;
        const Eigen::Vector2d& p = points[i];
        const Eigen::Vector3d Pc_norm(p[0], p[1], 1);

        const Eigen::Vector3d numerator = -hat(Pc_norm) * tca;
        const Eigen::Vector3d denominator = hat(Pc_norm) * Rca * Xa;
        s2 = numerator[0] / denominator[0];
        for (int i = 1; i < 3; ++i)
        {
            const double diff = s2 - numerator[i] / denominator[i];
            assert(abs(diff) < 1e-6);
        }
    }
    return poses[0] * (s2 * Xa);
}

Eigen::Vector3d FeatureTriangulation3(std::vector<Pose> poses, std::vector<Eigen::Vector2d> points)
{
    // Pw 需使用齐次坐标表示, 即(x/w, y/w, z/w, w/w),这是关键点
    // 最后的奇异值是4维的，最后一维是0空间，刚好有一个Ax = 0 的特解
    // 3X1 = 3X4 * 4X1
    // s * Pc = [Rcw | tcw] * Pw ==>
    // [Pc]x * [Rcw | tcw] * Pw = 0
    // 上述只能提供两个线性无关的方程
    const int num = poses.size();
    Eigen::MatrixXd A(2 * num, 4);
    for (int i = 0; i < num; ++i)
    {
        const Pose Tcw = poses[i].inverse();
        const Eigen::Matrix3d Rcw = Tcw.Rwc;
        const Eigen::Vector3d tcw = Tcw.twc;
        const Eigen::Vector2d& p = points[i];
        const Eigen::Vector3d Pc_norm(p[0], p[1], 1);
        Eigen::Matrix<double, 3, 4> B;
        B.block<3, 3>(0, 0) = Rcw;
        B.block<3, 1>(0, 3) = tcw;
        A.block<2, 4>(2 * i, 0) = (hat(Pc_norm) * B).block<2, 4>(0, 0);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "singularValue: " << svd.singularValues().transpose() << std::endl;
    Eigen::Vector4d P = svd.matrixV().col(3);
    return P.head(3) / P[3];
}

// OpenVins - 1D Depth Triangulation, No Good Actually!
Eigen::Vector3d FeatureTriangulation4(std::vector<Pose> poses, std::vector<Eigen::Vector2d> points)
{
    // s1 * Xa = s2 * Rac * Xc + tac
    // [Rac * Xc]x * s1 * Xa = [Rac * Xc]x * tac
    // (3X1).T * (3X1) = (1X1)
    // s1 {[Rac * Xc]x * Xa}.T * {[Rac * Xc]x * Xa}
    // = {[Rac * Xc]x * Xa}.T * [Rac * Xc]x * tac
    // 这个和FeatureTriangulation1其实是等价的
    const int num = poses.size();
    const Pose Twa = poses[0];
    const Pose Taw = Twa.inverse();
    const Eigen::Vector3d Xa{points[0][0], points[0][1], 1};
    double s1 = 0;
    for (int i = 1; i < num; ++i)
    {
        const Pose Tac = Taw * poses[i];
        const Eigen::Matrix3d Rac = Tac.Rwc;
        const Eigen::Vector3d tac = Tac.twc;
        const Eigen::Vector2d& p = points[i];
        const Eigen::Vector3d Xc(p[0], p[1], 1);

        const Eigen::Matrix3d Rac_Xc_hat = hat(Rac * Xc);
        const double numerator = (Rac_Xc_hat * Xa).transpose() * Rac_Xc_hat * tac;
        const double denominator = (Rac_Xc_hat * Xa).transpose() * Rac_Xc_hat * Xa;
        s1 += numerator / denominator;
    }
    s1 /= (num - 1);
    return poses[0] * (s1 * Xa);
}

// OpenVins - 3D Depth Triangulation!
Eigen::Vector3d FeatureTriangulation5(std::vector<Pose> poses, std::vector<Eigen::Vector2d> points)
{
    // Rcw * Pw + tcw = s * Xc
    // [Xc]x * Rcw * Pw = -[Xc]x * tcw
    const int num = poses.size();
    Eigen::MatrixXd A(3 * num, 3);
    Eigen::VectorXd b(3 * num, 1);
    for (int i = 0; i < num; ++i)
    {
        const Eigen::Matrix3d Rcw = poses[i].Rwc.transpose();
        const Eigen::Vector3d tcw = -Rcw * poses[i].twc;
        const Eigen::Vector2d& p = points[i];
        const Eigen::Vector3d Xc(p[0], p[1], 1);
        A.block(3 * i, 0, 3, 3) = hat(Xc) * Rcw;
        b.middleRows(3 * i, 3) = -hat(Xc) * tcw;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "singularValue: " << svd.singularValues().transpose() << std::endl;
    std::cout << "conditional number: " << svd.singularValues()[2] / svd.singularValues()[0] << std::endl;
    const Eigen::Vector3d Pw = A.colPivHouseholderQr().solve(b);
    return Pw;
}

int main()
{
    int featureNums = 20;
    int poseNums = 10;
    int diem = poseNums * 6 + featureNums * 3;
    double fx = 1.;
    double fy = 1.;
    Eigen::MatrixXd H(diem, diem);
    H.setZero();

    std::vector<Pose> camera_pose;
    double radius = 8;
    for (int n = 1; n < poseNums + 1; ++n)
    {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    // std::vector<Eigen::Vector3d> points;
    // j个特征点
    // J = poseNums * 6 + j * 3
    for (int j = 0; j < featureNums; ++j)
    {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(8., 10.);
        double tx = xy_rand(generator);
        double ty = xy_rand(generator);
        double tz = z_rand(generator);

        Eigen::Vector3d Pw(tx, ty, tz);
        // points.push_back(Pw);
        const int J = poseNums * 6 + j * 3;
        // i个相机pose
        // I = 6 * i
        std::vector<Pose> poses;
        std::vector<Eigen::Vector2d> points;
        for (int i = 0; i < poseNums; ++i)
        {
            const int I = 6 * i;

            Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
            Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

            double x = Pc.x();
            double y = Pc.y();
            double z = Pc.z();
            double z_2 = z * z;

            points.push_back({x / z, y / z});
            poses.push_back(camera_pose[i]);
            // std::cout << "iter " << i << (camera_pose[i] * Pc).transpose()
            // <<std::endl;

            Eigen::Matrix<double, 2, 3> jacobian_uv_Pc;
            jacobian_uv_Pc << fx / z, 0, -x * fx / z_2, 0, fy / z, -y * fy / z_2;
            // 关于地图点的导数
            Eigen::Matrix<double, 2, 3> jacobian_Pj = jacobian_uv_Pc * Rcw;

            // Eigen::Vector3d delta_t = Pw - camera_pose[i].twc * xy_rand(generator);
            // // 线性化点不一致使得0空间改变
            Eigen::Vector3d delta_t = Pw - camera_pose[i].twc;
            Eigen::Matrix<double, 3, 6> jacobian_Pc_Ti = Eigen::Matrix<double, 3, 6>::Zero();
            jacobian_Pc_Ti.block<3, 3>(0, 0) = -Rcw * hat(delta_t);
            jacobian_Pc_Ti.block<3, 3>(0, 3) = -Rcw;
            Eigen::Matrix<double, 2, 6> jacobian_Ti = jacobian_uv_Pc * jacobian_Pc_Ti;
            // ====================================
            // 对于单目BA，Ti与Tj之间没有约束，故雅可比矩阵为对角块矩阵
            // 对于VIO，由于新的残差 er, ev, ep由帧间约束确定，雅可比不再是对角块矩阵
            // 且由于重力方向可观测pitch, roll，尺度由IMU提供，因此零空间维度为4
            //    T0  T1  T2 ... P0 P1 P2 ...
            // r1
            // r2
            // r3
            // ====================================
            //
            // H矩阵需要计算第 I行第J列的组合值，即计算
            // H(I, I), H(I, J), H(J, I), H(J, J)
            H.block<6, 6>(I, I) += jacobian_Ti.transpose() * jacobian_Ti;
            H.block<6, 3>(I, J) += jacobian_Ti.transpose() * jacobian_Pj;
            H.block<3, 6>(J, I) += jacobian_Pj.transpose() * jacobian_Ti;
            H.block<3, 3>(J, J) += jacobian_Pj.transpose() * jacobian_Pj;
        }
        // Eigen::Vector3d PEw = FeatureTriangulation3(poses, points);
        Eigen::Vector3d PEw = FeatureTriangulation5(poses, points);
        std::cout << "true Pa = " << Pw.transpose() << std::endl;
        std::cout << "eval Pa = " << PEw.transpose() << std::endl;
        std::cout << std::endl;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "singular num = " << svd.singularValues().rows() << std::endl;
    Eigen::VectorXd s = svd.singularValues().tail(10);
    std::cout << s.transpose() << std::endl;
    int zero_count = 0;
    for (int i = 0; i < s.rows(); ++i)
    {
        if (s[i] < 1e-6)
        {
            zero_count++;
        }
    }
    std::cout << "Null space should be " << 7 << " counted is " << zero_count << std::endl;
    return 0;
}
