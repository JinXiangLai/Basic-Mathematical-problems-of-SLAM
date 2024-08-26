#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace std;

/***
* 车辆在利用隧道、车道线等信息约束横向平移的时候，
* 由于这些信息存在退化方向，车辆行进过程又不一定与退化方向平行，
* 因此需要找出退化方向，并消除平移增量在该方向的估计值，
* 然后再重新转回对应的坐标系，得到的结果才是正确的
***/

int main()
{
    Eigen::Matrix2d Rv0 = Eigen::Matrix2d::Identity();
    Eigen::Vector2d t_v0 = Eigen::Vector2d::Zero();

    // 修改退化方向以验证不同退化方向的基坐标系
    Eigen::Vector2d degraded_direction(1, std::sqrt(3));
    assert(degraded_direction != Eigen::Vector2d::Zero());
    degraded_direction.normalize(); // 归一化基向量

    // a * b = ax * bx + ay * by = 0
    Eigen::Vector2d vertical_direction = Eigen::Vector2d::Zero();
    if(degraded_direction[0] != 0){
        vertical_direction[1] = 1;
        vertical_direction[0] = -(degraded_direction[1] * vertical_direction[1]) / degraded_direction[0];
    } else{
        vertical_direction[0] = 1;
        vertical_direction[1] = -(degraded_direction[0] * vertical_direction[0]) / degraded_direction[1];
    }
    vertical_direction.normalize();
    std::cout << "degraded_direction: " << degraded_direction.transpose() << std::endl;
    std::cout << "vertical_direction: " << vertical_direction.transpose() << std::endl;
    std::cout << "degraded_direction * vertical_direction: " << degraded_direction.dot(vertical_direction) << std::endl;

    // 世界系下退化方向相关的新基表示
    Eigen::Matrix2d Rv0_d;
    Rv0_d.block<2, 1>(0, 0) = degraded_direction;
    Rv0_d.block<2, 1>(0, 1) = vertical_direction;

    // 用于测试的增量平移
    const Eigen::Vector2d Pv0(std::sqrt(3), 1);
    // 世界系下点变换到基坐标系下表示
    Eigen::Vector2d Pd = Rv0_d.transpose() * Pv0;
    // 消除退化方向平移增量
    Pd[0] = 0;
    // 退化基重新变换到世界下的平移增量
    const Eigen::Vector2d Pv0_ = Rv0_d * Pd;
    std::cout << "Pv0: " << Pv0.transpose() << std::endl;
    std::cout << "Pv0_: " << Pv0_.transpose() << std::endl;
    std::cout << "Pv0_.norm: " << Pv0_.norm() << std::endl;
    return 0;
}