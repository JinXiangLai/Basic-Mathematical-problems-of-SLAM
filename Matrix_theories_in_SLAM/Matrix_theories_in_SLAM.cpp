#include <iostream>
#include <Eigen/Dense>

using namespace std;

// Some basic konwledge of solving linear equations Ax = b
// In the update step of ESKF method, A is a [mxn] matrix, in which m > n;
// In the calculation of update step of Nolinear Optimization problem, A is a [nxn] matrix

Eigen::Matrix3d SymmetricMatrix(const Eigen::Vector3d &v){
    Eigen::Matrix3d m;
    m << 0, -v[2], v[1],
         v[2], 0, -v[0],
        -v[1], v[0], 0;
    return m;
}

Eigen::VectorXd SolveUpperTriangularMatrixEquation(const Eigen::MatrixXd &R, const Eigen::VectorXd &b){
    // Invertable matrix is needed by LU decomposition.
    if(R.rows() != b.rows() || R.cols() != b.rows() || !R.rows() || !b.rows() ){
        printf("[ERROR] R.rows = %ld & b.rows = %ld\n", R.rows(), b.rows());
        exit(-1);
    }
    Eigen::VectorXd res(b.rows());
    res.setZero();
    // Solve the equations from bottom to top and then from right to left.
    int col = R.cols() - 1;
    for(int i = R.rows() - 1; i >= 0 && col >= 0; --i, --col){
        double sum_from_right = 0;
        for(int j = R.cols() - 1; j > col; --j){
            sum_from_right += R.row(i)(j) * res(j);
        }
        const double product = b(i) - sum_from_right;
        res(i) = product / R.row(i)(col);
    }
    return res;
}

Eigen::VectorXd SolveLowerTriangularMatrixEquation(const Eigen::MatrixXd &L, const Eigen::VectorXd &b){
    // Invertable matrix is needed by LU decomposition.
    if(L.rows() != b.rows() || L.cols() != b.rows() || !L.rows() || !b.rows()){
        printf("[ERROR] L.rows = %ld & b.rows = %ld\n", L.rows(), b.rows());
        exit(-1);
    }
    Eigen::VectorXd res(b.rows());
    res.setZero();
    // Solve the equations from top to bottom and then from left to right
    int col = 0;
    for(int i = 0; i < L.rows(); ++i, ++col){
        double sum_from_front = 0;
        for(int j = 0; j < col; ++j){
            sum_from_front += L.row(i)(j) * res(j);
        }
        const double product = b(i) - sum_from_front;
        res(i) = product / L.row(i)(col);
    }
    return res;
}

int main(){
    // Start by solving the linear equations
    cout << std::fixed; // 不使用科学计数法
    cout.precision(2);
    cout << "\n******Solved full rank matrix A0*x0 = b0******\n" << endl;
    /**********
    * 3x + 5y + 6z = 10
    * 4x + 7y + 8z = 15
    * 6x + 4y + 6z = 9
    * 2x + 4y + 7z = 8;
    *********/
    Eigen::Matrix<double, 3, 3> A0;
    A0 << 3, 5, 6,
         4, 7, 8,
         6, 4, 6;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd0(A0, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d S0 = Eigen::Matrix3d::Zero();
    S0.diagonal() = svd0.singularValues();
    const Eigen::Matrix3d A0_inv = svd0.matrixV() * S0.inverse() * svd0.matrixU().transpose();
    Eigen::Matrix<double, 3, 1> b0;
    b0 << 31, 42, 32;
    cout << "A0 singular values: " << svd0.singularValues().transpose() << " Full rank" << endl;
    const Eigen::Vector3d x0 = (A0_inv * b0);
    cout << "x0 = A0_inv * b0 = " << x0.transpose() << " = [1,2,3]" << endl;
    cout << "U != V" << endl;
    cout << "A_0 U:\n" << svd0.matrixU() << endl;
    cout << "A_0 V:\n" << svd0.matrixV() << endl;
    cout << "U * U':\n" << svd0.matrixU() * svd0.matrixU().transpose() << endl;
    cout << "V * V':\n" << svd0.matrixV() * svd0.matrixV().transpose() << endl;
    cout << "U * V':\n" << svd0.matrixU() * svd0.matrixV().transpose() << endl;
    cout << "transpose then invert = invert then transpose:" << endl;
    cout << "A0'.inv:\n" << A0.transpose().inverse() << endl;
    cout << "A_0.inv':\n" << A0.inverse().transpose() << endl;

    // Solve Ax = b by [b]x * Ax = 0, which will make scale unobservable
    cout << "\n******Solve (A0*x0 = b0) by ([b0]x *A0*x0 = 0), which will make the scale unobservable******\n" << endl;
    Eigen::Matrix<double, 3, 3> C1;
    C1 = SymmetricMatrix(b0) * A0;
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd1(C1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    cout << "singular values: " << svd1.singularValues().transpose() << " Rank = 2" << endl;
    const Eigen::Vector3d res0 = svd1.matrixV().col(2);
    cout << "x0 is the last col of V: " << res0.transpose() << endl;
    cout << "scale = " << x0[0] / res0[0] << endl;
    // 行列式不为0即可逆
    cout << "A0 determinant: " << A0.determinant() << " != 0, invertable" << endl;
    cout << "C1 determinat: " << C1.determinant() << " == 0, uninvertable" << endl;

    
    cout << "\n******Solve A0*x0 = b0 by QR decomposition, A0 must be invertable******\n" << endl;
    // QR分解：
    Eigen::HouseholderQR<Eigen::Matrix3d> qr(A0);
    Eigen::Matrix3d Q = qr.householderQ();
    cout << "Q * Q':\n" << Q * Q.transpose() << endl;
    Eigen::Matrix3d R = Q.transpose() * A0;
    cout << "R:\n" << R << endl;
    cout << "Q * R = A0:\n" << Q * R << endl;
    const Eigen::Vector3d y0 = Q.transpose() * b0;
    // R*x0 = y0
    Eigen::Vector3d res1 = SolveUpperTriangularMatrixEquation(R, y0);
    cout << "x0 - [res1 by QR]: " << (x0 - res1).transpose() << " = vector 0" << endl;

    
    // 验证地图点三角化的零空间维度
    cout << "\n******** Verify the nullspace dimension of triangulating points(4D) ******\n";
    Eigen::Matrix3d Rcw = Eigen::AngleAxisd(M_PI * 0.3, Eigen::Vector3d(1, 2, 3)).toRotationMatrix();
    Eigen::Vector3d Pcw(1, 2, 3);
    Eigen::Vector3d pw(3, 4, 5);
    // Tc0 = [I | 0]
    const Eigen::Vector3d obv0 = pw / pw[2];
    const Eigen::Vector3d pc = Rcw * pw + Pcw;
    // Tc1 = [Rcw | Pcw]
    cout << "pc: " << pc.transpose() << endl;
    const Eigen::Vector3d obv1 = pc / pc[2];
    // [R | t] * Pw = s * obv
    // [obv]x * [R | t] * Pw = 0
    Eigen::Matrix<double, 6, 4> A1;
    Eigen::Matrix<double, 3, 4> Tc0;
    Tc0.setZero();
    Tc0.block(0, 0, 3, 3).setIdentity();
    A1.block(0, 0, 3, 4) = SymmetricMatrix(obv0) * Tc0;
    Eigen::Matrix<double, 3, 4> Tc1;
    Tc1.block(0, 0, 3, 3) = Rcw;
    Tc1.block(0, 3, 3, 1) = Pcw;
    A1.block(3, 0, 3, 4) = SymmetricMatrix(obv1) * Tc1;
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd3(A1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector4d singular_value = svd3.singularValues();
    cout << "singular values: " << singular_value.transpose() << endl;
    int rank0 = 0; 
    for(int i = 0; i < singular_value.rows(); ++i){
        if(singular_value(i) > 1e-6){
            ++rank0;
        }
    }
    cout << "The rank of triangulating a 3D point(represent by 4D) is: " << rank0 << endl;
    cout << "Nullspace dimension: " << (A1.cols() - rank0) << endl;
    Eigen::Vector4d p3d0 = svd3.matrixV().col(3);
    p3d0 /= p3d0[3];
    cout << "3D map point0: " << p3d0.head(3).transpose() << " = [3, 4, 5]" << endl;

    // [R | t] * p = obv
    cout << "\n***Use pseudo inverse to triangulate a 3D point. Can't succeed!!!***\n";
    Eigen::Matrix<double, 6, 4> A2;
    A2.setZero();
    A2.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    A2.block(3, 0, 3, 3) = Rcw;
    A2.block(3, 3, 3, 1) = Pcw;
    Eigen::Matrix<double, 6, 1> b2;
    b2.head(3) = obv0; // *s
    b2.tail(3) = obv1; // *s
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd4(A2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 6> U2 = svd4.matrixU();
    Eigen::Matrix<double, 4, 4> V2 = svd4.matrixV();
    Eigen::Matrix<double, 6, 4> S2;
    S2.setZero();
    S2.diagonal() = svd4.singularValues();
    cout << "svd4.singularValues(): " << svd4.singularValues().transpose() << endl;
    Eigen::Matrix<double, 4, 6> S2_inv;
    S2_inv.setZero();
    for(int i = 0; i < S2.diagonal().size(); ++i){
        if(S2.diagonal()[i] < 1e-6){
            S2.diagonal()[i] = 1e-6;
        }
        S2_inv.diagonal()[i] = 1.0 / S2.diagonal()[i];
    }
    cout << "S1:\n" << S2 << endl;
    cout << "S1_inv:\n" << S2_inv << endl;
    // A = U * ∑ * V' ==> A.inv = V * ∑.inv * U'
    Eigen::Matrix<double, 4, 6> A2_inv = V2 * S2_inv * U2.transpose();
    cout << "A2_inv * A2:\n" << A2_inv * A2 << endl;
    cout << "A2 * A2_inv:\n" << A2 * A2_inv << endl;
    Eigen::Vector4d p3d1 = A2_inv * b2;
    cout << "p3d1: " << p3d1.transpose() << endl;
    p3d1 /= p3d1[3];
    cout << "3D map point1: " << p3d1.head(3).transpose() << endl;

    // LU Lower(or Upper) triangular matrix decomposition
    cout << "\n****** Use LU decomposition to solve Ax=b, A must be invertable.******\n";
    Eigen::Matrix3d A3;
    A3 << 4, 5, 6,
          0, 2, 3,
          0, 0, 1;
    Eigen::Vector3d b3;
    b3 << 28, 7, 1;
    const Eigen::Vector3d res2 = SolveUpperTriangularMatrixEquation(A3, b3);
    cout << "res2: " << res2.transpose() << " = [3, 2, 1]" << endl;
    Eigen::Matrix3d A3_1;
    A3_1 << 1, 0, 0,
            2, 3, 0,
            4, 5, 6;
    Eigen::Vector3d b3_1;
    b3_1 << 1, 8, 32;
    const Eigen::Vector3d res2_1 = SolveLowerTriangularMatrixEquation(A3_1, b3_1);
    cout << "res2_1: " << res2_1.transpose() << " = [1, 2, 3]"<< endl;
    

    // ill-conditioned matrix
    cout << "\n******Verify that the (H+λI) is unill-conditioned******\n";
    Eigen::Matrix<double, 2, 2> A4;
    Eigen::JacobiSVD<Eigen::Matrix2d> svd5(A4, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector2d singular4 = svd5.singularValues();
    A4 << 400, -201,
         -800, 201;
    cout << "[ill] Det(A4): " << A4.determinant() << endl;
    cout << "singular values of A4: " << singular4.transpose() << endl;;
    cout << "conditional num of A4: " << singular4.head(1)(0) / singular4.tail(1)(0) << endl;
    A4 += Eigen::Matrix2d::Identity();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd5_1(A4, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector2d singular4_1 = svd5_1.singularValues();
    cout << "[ill] Det(A4+I): " << A4.determinant() << endl;
    cout << "singular values of (A4+I): " << singular4_1.transpose() << endl;;
    cout << "conditional num of (A4+I): " << singular4_1.head(1)(0) / singular4_1.tail(1)(0) << endl;

    // when residual's dimension m < state vector dimension
    // e.g    x0  x1  x2
    //    r0  J0  J1  J2
    Eigen::Matrix<double, 1, 3> J1(1, 2, 3);
    Eigen::Matrix3d H1 = J1.transpose() * J1;
    cout << "Hessian:\n" << H1 << endl;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd6(H1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular6 = svd6.singularValues();
    cout << "m < n, singular values of Hessian matrix : " << singular6.transpose() << endl;
    H1 += Eigen::Matrix3d::Identity();
    cout << "H:\n" << H1 << endl;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd6_1(H1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular6_1 = svd6_1.singularValues();
    cout << "m < n, [H+I] singular values: " << singular6_1.transpose() << endl;
    return 0; 
}