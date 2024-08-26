#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;

/******** 验证在解SLAM的非线性优化问题时，极小值与初值密切相关******************
* 由于SLAM问题的状态量是多维向量，因此其问题求解模型抽象为多元二次方程
* 使 F = w1*(f(x) - r1)^2 + w2*(f(x) - r2)^2 值最小的解X为最优；
* 其中X为多维状态向量
*********************************************************************************/

constexpr int kConstraintNum = 2;
Eigen::Matrix<double, kConstraintNum, 1> kWeight;

void GetJacobianAndResidualFull(const Eigen::Vector2d& x, const vector<double> &obv, Eigen::Matrix<double, kConstraintNum, 2> &Jfull, 
                                    Eigen::Matrix<double, kConstraintNum, 1> &residual);

void UpdateX(const Eigen::Vector2d &delta_x, Eigen::Vector2d& x);

double CalculateCost(const Eigen::Vector2d &x, const vector<double> &obv);

// 为了避免构造出线性相关的雅可比矩阵(即线性方程组系数)，这里构建多个不同的量测函数
double GetObvFunctionValue(const Eigen::Vector2d &x, const int index);

int main() {
    // 预设初值和权重值
    // Eigen::Vector2d x(-0.1, -0.1);
    Eigen::Vector2d x(0.1, 0.1);
    kWeight << 1, 1;

    // 产生真实值x的量测数据
    const Eigen::Vector2d x_t{2, 2};
    vector<double> obv{GetObvFunctionValue(x_t, 0), GetObvFunctionValue(x_t, 1)};
    cout << "obv value: ";
    for(double d : obv) {
        cout << d << " ";
    }
    cout << endl << endl;

    double lastCost = __DBL_MAX__;
    double curCost = CalculateCost(x, obv);
    int step = 0;
    cout << "step " << step << " cost: " << curCost  << "\n====================\n" << endl;
    // 高斯-牛顿法迭代求解
    while (abs(lastCost - curCost) > 1e-4)
    {
        lastCost = curCost;
        ++step;
        Eigen::Matrix<double, kConstraintNum, 2> Jfull = Eigen::Matrix<double, kConstraintNum, 2>::Zero();
        Eigen::Matrix<double, kConstraintNum, 1> residual = Eigen::Matrix<double, kConstraintNum, 1>::Zero();
        
        // 构建残差和雅可比
        GetJacobianAndResidualFull(x, obv, Jfull, residual);

        // 构建H*Δx = g, 求解delta_x增量
        Eigen::Matrix<double, kConstraintNum, kConstraintNum> W;
        W.setZero();
        W.diagonal() = kWeight;
        const Eigen::Matrix2d H = Jfull.transpose() * W * Jfull;
        const Eigen::Vector2d b = -Jfull.transpose() * W * residual;
        const Eigen::Vector2d delta_x = H.colPivHouseholderQr().solve(b);
        
        // 更新状态量估计值
        UpdateX(delta_x, x);
        curCost = CalculateCost(x, obv);
        cout << "step " << step << " cost: " << curCost << endl << endl;
    }
    
    cout << "The best x | cost: " << x.transpose() << " | " << curCost << endl << endl;
    return 0;
}

void GetJacobianAndResidualFull(const Eigen::Vector2d& x, const vector<double> &obv, Eigen::Matrix<double, kConstraintNum, 2> &Jfull, 
                                    Eigen::Matrix<double, kConstraintNum, 1> &residual) {
    // 计算当前step的残差值
    for(int i = 0; i < kConstraintNum; ++i) {
        residual[i] = GetObvFunctionValue(x, i) - obv[i];
    }

    // 计算当前step的雅可比
    for(int i = 0; i < kConstraintNum; ++i) {
        // 注意：对于SLAM问题中，由于每个constraint不可能与所有状态量产生关联，
        // 并且，每个constraint的类型也不是完全一致的，
        // 所以其J矩阵的性质为：
        // 一是每一行会产生很多0项；
        // 二是J矩阵的每一行的(系数)值基本都是差异较大的，即雅可比矩阵线性无关
        // 这里我们只是简化考虑了一个，但仍需要有线性无关的雅可比才可以令H*Δx = g有特解
        switch(i) {
            case 0:
                Jfull.row(0) << 2*x[0], 2*x[1];
                break;
            case 1:
                Jfull.row(1) << 2*x[0], -2*x[1];
                break;
            default:
                cerr << "[Error] index should be in [0, 1]!" << endl;
                exit(-1);
        }
    }
    return;                                  
}

void UpdateX(const Eigen::Vector2d &delta_x, Eigen::Vector2d& x) {
    x += delta_x;
}

double CalculateCost(const Eigen::Vector2d &x, const vector<double> &obv) {
    double res = 0;
    for(int i = 0; i < kConstraintNum; ++i) {
        res += kWeight[i] * pow(GetObvFunctionValue(x, i) - obv[i], 2);
    }
    return res;
}

double GetObvFunctionValue(const Eigen::Vector2d &x, const int index) {
    // 构建多个线性无关的约束问题
    switch(index) {
        case 0:
            return pow(x[0], 2) + pow(x[1], 2);
            break;
        case 1:
            return pow(x[0], 2) - pow(x[1], 2);
            break;
        default:
            cerr << "[Error] index should be in [0, 1]!" << endl;
            exit(-1);
    }

}
