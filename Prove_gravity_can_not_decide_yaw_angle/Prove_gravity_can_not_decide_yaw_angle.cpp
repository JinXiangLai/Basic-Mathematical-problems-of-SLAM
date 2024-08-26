#include <iostream>

#include <Eigen/Dense>

using namespace std;

int main()
{
    const double gravityVal = 9.81;
    const Eigen::Vector3d g(0, 0, -gravityVal);
    // gravity in IMU coordinate
    Eigen::Vector3d gInAcc(1, 2, 3);
    gInAcc.normalize();
    gInAcc *= gravityVal;

    // measurement of gravity with IMU
    const Eigen::Vector3d gMeasurementInAcc(-gInAcc);

    Eigen::Vector3d rot_axis = gInAcc.cross(g);
    rot_axis.normalize();
    const double angle = std::acos((gInAcc.dot(g)) / (gInAcc.norm() * g.norm()));

    const Eigen::Matrix3d Rga = Eigen::AngleAxisd(angle, rot_axis).matrix();
    const Eigen::Matrix3d Rag = Rga.transpose();
    Eigen::Vector3d ypr = Rag.eulerAngles(2, 1, 0);
    
    Eigen::IOFormat fmt(2);
    std::cout << std::fixed; // Scientific numeration output is prohibited

    std::cout << "\n===Rotate g to gInAcc===\n";
    std::cout << "gInAcc = " << gInAcc.transpose() << std::endl;
    for (int i = 1; i <= 5; ++i)
    {
        ypr[0] *= i;
        const Eigen::Matrix3d Rag2 =
            (Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX()))
                .matrix();
        const Eigen::Vector3d gInAcc2 = Rag2 * g;
        std::cout << "[" << i << " time gInAcc2 - gInAcc]: " << (gInAcc2 - gInAcc).transpose().format(fmt) << std::endl;
    }

    std::cout << "\n===Rotate gInAcc to g===\n";
    ypr = Rga.eulerAngles(2, 1, 0);
    std::cout << "g = " << g.transpose() << std::endl;
    for (int i = 1; i <= 5; ++i)
    {
        ypr[0] *= i;
        const Eigen::Matrix3d Rga2 =
            (Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX()))
                .matrix();
        const Eigen::Vector3d g2 = Rga2 * gInAcc;
        std::cout << "[" <<i << " time g2 - g]: " << (g2 - g).transpose().format(fmt) << std::endl;
    }

    return 0;
}
