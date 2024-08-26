#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;

/******** 验证在解SLAM的非线性或者线性最小二乘时，结果会向权值大的等式靠拢******************
* 最小二乘的本质也是简单的线性方程求解，只是其求解的问题为：
* 使 F = w1*(f(x) - r1)^2 + w2*(f(x) - r2)^2 + w3*(f(x) - r3)^2 值最小的解X为最优；
* 可以看出，每个残差项(f(x) - ri)^2都是非线性的二次项；
* 令f(x) = x+y^2，一般利用高斯-牛顿法求解X(x, y)，易得：
* 残差关于状态量的雅可比 J = [1, 2*y]，那么每个残差构建的等式为：
* J' * W * J * Δx = -J' * W * r
* 则理论上，最优解X受权重W大的影响越大。
* 那么可构建求解满足F方程值最小的X(x, y)的解，这里要验证，解X会往权重wi大的预设解集(x, y)靠拢
*********************************************************************************/

constexpr int kConstraintNum = 6;
Eigen::Matrix<double, kConstraintNum, 1> kWeight;

void GetJacobianAndResidualFull(const Eigen::Vector2d& x, const vector<double> &obv, Eigen::Matrix<double, kConstraintNum, 2> &Jfull, 
                                    Eigen::Matrix<double, kConstraintNum, 1> &residual);

void UpdateX(const Eigen::Vector2d &delta_x, Eigen::Vector2d& x);

double CalculateCost(const Eigen::Vector2d &x, const vector<double> &obv);

// 为了避免构造出线性相关的雅可比矩阵(即线性方程组系数)，这里构建多个不同的量测函数
double GetObvFunctionValue(const Eigen::Vector2d &x, const int index);

int main() {
    // 预设初值和权重值
    Eigen::Vector2d x(0.1, 0.1);
    kWeight << 100, 1, 1, 
               100, 1, 1;

    // 根据预设的状态量xi给定相应的量测值，至少需给每个预设解提供不少于2个的线性无关方程
    const Eigen::Vector2d x0{2, 3}, x1{2+2, 3+2}, x2{2+3, 3+3};
    vector<double> obv{GetObvFunctionValue(x0, 0), GetObvFunctionValue(x1, 1), GetObvFunctionValue(x2, 2),
                        GetObvFunctionValue(x0, 3), GetObvFunctionValue(x1, 4), GetObvFunctionValue(x2, 5)};
    cout << "obv value: ";
    for(double d : obv) {
        cout << d << " ";
    }
    cout << endl << endl;

    double lastCost = __DBL_MAX__;
    double curCost = CalculateCost(x, obv);
    int step = 0;
    cout << "step " << step << " cost: " << curCost << endl << endl;
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
    
    cout << "The best x | cost " << x.transpose() << " | " << curCost << endl << endl;
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
                Jfull.row(0) << 1, 1.5;
                break;
            case 1:
                Jfull.row(1) << 2.5, 1.1;
                break;
            case 2:
                Jfull.row(2) << 3.1, 2.2;
                break;
            case 3:
                Jfull.row(3) << -3.1, 2.5;
                break;
            case 4:
                Jfull.row(4) << -1, 4;
                break;
            case 5:
                Jfull.row(5) << 4.3, -1;
                break;
            default:
                break;
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
            return x[0] + 1.5 * x[1];
            break;
        case 1:
            return 2.5 * x[0] + 1.1 * x[1];
            break;
        case 2:
            return 3.1 * x[0] + 2.2 * x[1];
            break;
        case 3:
            return -3.1 * x[0] + 2.5 * x[1];
            break;
        case 4:
            return -x[0] + 4 * x[1];
            break;
        case 5:
            return 4.3 * x[0] - x[1];
            break;
        default:
            break;
    }
    cout << "[Error] index should be in [0, 5]!" << endl;
    exit(-1);
}
