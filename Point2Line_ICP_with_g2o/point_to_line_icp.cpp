#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>

#include <sophus/se3.hpp>

#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>

#include "g2oTypes.hpp"

/***********
* 对于P-L ICP，经过分析：
* 平移自由度，只有在退化方向上的平移是不可观的；
* 姿态pitch、roll、yaw自由度可以完全约束，与车位角点等观测量是否是在一个平面上无关
************/

const double kPi = 3.1415926;
const double kDegree2Rad = kPi / 180;
const double kRad2Degree = 180 / kPi;
const Sophus::SE3d kTw;
const double roll = 5 * kDegree2Rad * 0;
const double pitch = -25 * kDegree2Rad * 0;
const double yaw = 15 * kDegree2Rad;


const Eigen::Quaterniond Qwb = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * \
Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * \
Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
const Eigen::Vector3d twb(-0.5, -0.2, -0.1);
const Sophus::SE3d Twb(Qwb, twb);
const Sophus::SE3d Tbw(Twb.inverse());


// 生成匹配像素点
void generatePoints(Eigen::MatrixXd& linePoints1, Eigen::MatrixXd& linePoints2)
{
	double h = 1.2;
	double w = 1.5;
	// {col[i], col[i+1]}构成一条直线
	const int lineNum = 3;
	linePoints1.resize(3, lineNum * 2);
	int col = 0;
	for (int i = 1; i <= lineNum; ++i) {
		double hi = h * i;
		for (int j = 1; j <= 2; ++j) {
			double wj = w * j;
			linePoints1.col(col++) = Eigen::Vector3d(hi, wj, static_cast<double>(std::rand() % 10));
			// linePoints1.col(col++) = Eigen::Vector3d(hi, wj, -0.38);
		}
	}

	std::cout << "linePoints1:\n" << linePoints1 <<std::endl;

	linePoints2.resize(linePoints1.rows(), linePoints1.cols());
	for (int i = 0; i < linePoints1.cols(); ++i) {
		const Eigen::Vector3d& pw = linePoints1.col(i);
		Eigen::Vector3d pb = Tbw * pw;
		linePoints2.col(i) = pb;
	}
}

int main(int argc, char** argv) {
	int its = 6;
	if (argc > 1) {
		std::string its2(argv[1]);
		its = std::stoi(its2);
	}


	// ============================================================ //
	// 这个Map类型使用需要研究一下
	typedef typename Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::AlignedMapType JacobianXiOplusType;
	// 还需要把B映射到一块内存
	JacobianXiOplusType B(0, 3, 3);
	double dataB[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	JacobianXiOplusType C(dataB, 3, 3);
	std::cout << "B.size: " << B.size() << std::endl;
	std::cout << "C:\n" << C << std::endl;
	// B = C; // 无法赋值
	// 重新映射B
	new (&B) JacobianXiOplusType(dataB, 3, 3);
	std::cout << "B:\n" << B << std::endl;
	B.setZero();
	std::cout << "C when zero B:\n" << C << std::endl;
	// ============================================================ //

	Eigen::MatrixXd linePoints1, linePoints2;
	generatePoints(linePoints1, linePoints2);

	const double roll2 = 3 * kDegree2Rad; // 5 * 0
	const double pitch2 = -1 * kDegree2Rad; // -25 * 0
	const double yaw2 = 12 * kDegree2Rad; // 15
	const Eigen::Quaterniond Qwb2 = Eigen::AngleAxisd(yaw2, Eigen::Vector3d::UnitZ()) * \
		Eigen::AngleAxisd(pitch2, Eigen::Vector3d::UnitY()) * \
		Eigen::AngleAxisd(roll2, Eigen::Vector3d::UnitX());
	const Eigen::Vector3d twb2(-0.4, 150000, 0.1); // (-0.5, -0.2, -0.1)
	const Sophus::SE3d Twb2(Qwb2, twb2);

	// Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	// solver->setUserLambdaInit(1.);
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	// Step 2: 初始化优化器的节点和边
	VertexEigenPose* T_wb = new VertexEigenPose(Qwb2, twb2);
	T_wb->setId(0);
	T_wb->setFixed(false);
	optimizer.addVertex(T_wb);

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0., 0.01);


	std::vector<EdgePoint2LineICP*> es;
	for (int i = 0; i < linePoints1.cols() / 2; ++i) {
		// 添加测量噪声用
		double noise = distribution(generator);
		Eigen::Vector3d vecNoise(noise, noise, noise);

		// 取出一条直线
		const Eigen::Vector3d pw1 = linePoints1.col(2 * i);
		const Eigen::Vector3d pw2 = linePoints1.col(2 * i + 1);

		// 添加2个点到直线距离的优化项
		const Eigen::Vector3d pb1 = linePoints2.col(2 * i); // + vecNoise;
		EdgePoint2LineICP *e1 = new EdgePoint2LineICP(pw1, pw2, pb1);
		e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		// 默认信息矩阵值是0,所以必须给定
		e1->setInformation(Eigen::Matrix3d::Identity());
		optimizer.addEdge(e1);

		const Eigen::Vector3d pb2 = linePoints2.col(2 * i + 1); // + vecNoise;
		EdgePoint2LineICP *e2 = new EdgePoint2LineICP(pw1, pw2, pb2);
		e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
		e2->setInformation(Eigen::Matrix3d::Identity());
		optimizer.addEdge(e2);

		es.push_back(e1);
		es.push_back(e2);
	}

	// Step 3: 执行优化
	optimizer.setVerbose(true);
	optimizer.initializeOptimization(0);


	optimizer.computeActiveErrors();
	std::cout << "initial optimizer.chi2(): " << optimizer.chi2() << std::endl;

	

	std::cout<<"====== Nullspace ======\n";
	Eigen::MatrixXd J(3 * es.size(), 6);
	for(size_t i=0; i<es.size(); ++i){
		J.block<3, 6>(i*3, 0) = es[i]->GetJacobian();
	}
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(J.transpose()*J, Eigen::ComputeFullU | Eigen::ComputeFullV);
	// 多少个为0就说明多少个量不可观，即系统自由度
	std::cout << "singularValues: " << svd.singularValues().transpose() << std::endl;

	optimizer.optimize(its);

	VertexEigenPose* Twb_recovery = static_cast<VertexEigenPose*>(optimizer.vertex(0));
	Sophus::SE3d Twb_final(Twb_recovery->estimate().qrb, Twb_recovery->estimate().trb);

	std::cout << "\n========\nCompare euler angles[y p r] and translation after optimizating " << its << " times:\n";
	std::cout << "Twb_true: " << "[" << Twb.rotationMatrix().eulerAngles(2, 1, 0).transpose() * kRad2Degree << "], [" <<
		Twb.translation().transpose() << "]\n";

	std::cout << "Twb_before opt:" << "[" << Twb2.rotationMatrix().eulerAngles(2, 1, 0).transpose() * kRad2Degree << "], [" <<
		Twb2.translation().transpose() << "]\n";

	std::cout << "Twb_after opt:" << "[" << Twb_final.rotationMatrix().eulerAngles(2, 1, 0).transpose() * kRad2Degree << "], [" <<
		Twb_final.translation().transpose() << "]\n";

	return 0;
}