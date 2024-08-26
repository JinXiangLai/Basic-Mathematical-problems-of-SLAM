#include <iostream>
#include <vector>
#include <stdio.h>

#include <Eigen/Dense>

using namespace std;

constexpr int kPointNum = 9;
constexpr double kDeg2Rad = M_PI / 180;
constexpr double kRad2Deg = 180 / M_PI;

Eigen::Matrix3d SymmetricMatrix(const Eigen::Vector3d &v){
    Eigen::Matrix3d m;
    m << 0, -v[2], v[1],
         v[2], 0, -v[0],
        -v[1], v[0], 0;
    return m;
}

Eigen::Vector3d SymmetricMatrix2Vector(const Eigen::Matrix3d &m){
    return Eigen::Vector3d(m.row(2)[1], m.row(0)[2], m.row(1)[0]);
}

Eigen::Vector3d Rot2RPY(const Eigen::Matrix3d &R){
    const Eigen::Quaterniond q(R);
    double q0 = q.w();
    double q1 = q.x();
    double q2 = q.y();
    double q3 = q.z();

    double	q11 = q0*q0, q12 = q0*q1, q13 = q0*q2, q14 = q0*q3, 
            q22 = q1*q1, q23 = q1*q2, q24 = q1*q3,     
            q33 = q2*q2, q34 = q2*q3,  
            q44 = q3*q3;
    Eigen::Vector3d rpy;
    rpy(0) = atan2(-2*(q24-q13), q11-q22-q33+q44);//roll
    rpy(1) = asin(2*(q34+q12));// pitch
    rpy(2) = atan2(-2*(q23-q14), q11-q22+q33-q44);//yaw
    return rpy;
}


void GeneratePointSet(Eigen::Matrix<double, 3, kPointNum> &srcPoints, Eigen::Matrix<double, 3, kPointNum> &tarPoints, 
        const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &Pcw){
    srcPoints.col(0) << 1.0, 1.5, 2.0;
    tarPoints.col(0) = Rcw * srcPoints.col(0) + Pcw;
    for(int i = 1; i < kPointNum; ++i){
        srcPoints.col(i) = srcPoints.col(i - 1) + Eigen::Vector3d::Random();
        tarPoints.col(i) = Rcw * srcPoints.col(i) + Pcw;
    }
    cout << "src points z:\n" << srcPoints.row(2) << endl;
    cout << "tar points z:\n" << tarPoints.row(2) << endl;
    
    // normlize
    for(int i = 0; i < kPointNum; ++i){
        srcPoints.col(i) /= srcPoints.col(i)[2];
        tarPoints.col(i) /= tarPoints.col(i)[2];
    }
    return;
}

bool CalculateEssentialMatrix(Eigen::Matrix<double, 3, kPointNum> &normPoints0, Eigen::Matrix<double, 3, kPointNum> &normPoints1, 
                                Eigen::Matrix3d &E){
    // R * p0 + t = s * p1
    // t^*R * p0 = s * t^ * p1 // 平行向量叉积=0
    // p1' * E * p0 = 0 // 垂直向量点积=0
    // A * e = 0
    // 若要使用8点法求解，必须使得E矩阵最后一个元素即E[2][2] = e[8] = 常数
    Eigen::MatrixXd A(kPointNum, 9);
    A.setZero();
    for(int i = 0; i < kPointNum; ++i) {
        const Eigen::Vector3d p0 = normPoints0.col(i);
        const Eigen::Vector3d p1 = normPoints1.col(i);
        const double x0 = p0(0), y0 = p0(1), z0 = p0(2),
                     x1 = p1(0), y1 = p1(1), z1 = p1(2);
        A.row(i) << x1*x0, x1*y0, x1,
                    y1*x0, y1*y0, y1,
                    x0, y0, 1;
    }
    // A * e = 0
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::MatrixXd eigenValue = svd.singularValues();
    cout << "eigenValue of matrix A: " << eigenValue.transpose() << endl;
    int rank = 0;
    for(int i = 0; i < eigenValue.size(); ++i){
        if(eigenValue(i) > 1e-6){
            ++rank;
        }
    }
    if(rank < 8){
        cout << "rank of A matrix = " << rank << " <8, can't compute matrix E" << endl;
        return false; 
    }
    Eigen::Matrix<double, 9, 1> e = svd.matrixV().col(8);
    cout << "e: " << e.transpose() << endl;
    for(int i = 0; i < 3; ++i){
        E.row(i) = e.middleRows(3 * i, 3);
    }
    return true;
}

void DecomposeEssentialMatrix(const Eigen::Matrix3d &E, vector<Eigen::Matrix3d> &Rs, vector<Eigen::Vector3d> &ts){
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S.diagonal() = svd.singularValues();
    cout << "Eigen values of matrix E: " << S.diagonal().transpose() << endl;
    const Eigen::Matrix3d Rz = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d _Rz = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    ts.resize(2);
    Rs.resize(2);
    ts[0] = SymmetricMatrix2Vector(U * Rz * S * U.transpose());
    ts[1] = SymmetricMatrix2Vector(U * _Rz * S * U.transpose());
    Rs[0] = U * Rz.transpose() * V.transpose();
    Rs[1] = U * _Rz.transpose() * V.transpose();
}

void ChooseRightPose(const vector<Eigen::Matrix3d> &Rs, const vector<Eigen::Vector3d> &ts, const Eigen::Vector3d &pw,
                        Eigen::Matrix3d &Rt, Eigen::Vector3d &tt){
    int Ridx = -1;
    int Pidx = -1;
    for(int i = 0; i < 2; ++i){
        const Eigen::Matrix3d &R = Rs[i];
        const Eigen::Vector3d &t = ts[i];
        const Eigen::Vector3d pc = R * pw + t;
        printf("[%d][%d], pc.z = %f\n", i, i, pc[2]);
        if(pc[2] > 0){
            Rt = R;
            tt = t;
        }
    }
}


int main(){
    const Eigen::Matrix3d Rcw = Eigen::AngleAxisd(30 * kDeg2Rad, Eigen::Vector3d::UnitZ()) * 
                                Eigen::AngleAxisd(5 * kDeg2Rad, Eigen::Vector3d::UnitY()) * 
                                Eigen::AngleAxisd(3 * kDeg2Rad, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Vector3d Pcw(1.0, 2.0, 0.1);
    Eigen::Matrix<double, 3, kPointNum> srcPoints, tarPoints;
    GeneratePointSet(srcPoints, tarPoints, Rcw, Pcw);
    Eigen::Matrix3d E;
    if(!CalculateEssentialMatrix(srcPoints, tarPoints, E)){
        return -1;
    }
    vector<Eigen::Matrix3d> Rs;
    vector<Eigen::Vector3d> ts;
    DecomposeEssentialMatrix(E, Rs, ts);
    Eigen::Matrix3d Rt;
    Eigen::Vector3d tt;
    ChooseRightPose(Rs, ts, srcPoints.col(0), Rt, tt);
    cout << "\n****** Check Result ******:" << endl;
    cout << "Rt * Rcw':\n" << Rt * Rcw.transpose() << endl << endl;
    cout << "tt x Pcw: " << (tt.cross(Pcw)).transpose() << endl;
    return 0;
}