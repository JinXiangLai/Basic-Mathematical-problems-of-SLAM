#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>

#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>

#include "g2oTypes.hpp"

typedef Eigen::Matrix<double, 3, 3>  Matrix3d;
#define USE_INVERSE_DEPTH 1

const double kPi = 3.1415926;
const double kDegree2Rad = kPi / 180;
const double kRad2Degree = 180 / kPi;
const Sophus::SE3d kTw;
const double roll = 5 * kDegree2Rad;
const double pitch = -25 * kDegree2Rad;
const double yaw = 15 * kDegree2Rad;


const Eigen::Quaterniond Qwb = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * \
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * \ 
                            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
const Eigen::Vector3d twb(-0.5, -0.2, -2.0);
const Sophus::SE3d Twb(Qwb, twb);
const Sophus::SE3d Tbw(Twb.inverse());

#if USE_INVERSE_DEPTH
    const Eigen::Quaterniond Qwc = Eigen::AngleAxisd(yaw+5, Eigen::Vector3d::UnitZ()) * \
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * \ 
                            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    const Eigen::Vector3d twc(-0.8, -0.8, -2.4);
    const Sophus::SE3d Twc(Qwc, twc);
    const Sophus::SE3d Tcw(Twc.inverse());
#endif

// 生成匹配像素点
void generatePoints(Eigen::MatrixXf& pixPoints1, Eigen::MatrixXf& pixPoints2,
                    Eigen::MatrixXf& mapPoints, Eigen::MatrixXf& pixPoints3)
{
    //选取相机1的n个像素点
    CameraModel cam;
    int h = cam.img_height/4;
    int w = cam.img_width/4;
    pixPoints1.resize(3, 9);
    // 生成相机1像素点
    for(int i=1; i<=3; ++i){
        int hi = h * i;
        for(int j=1; j<=3; ++j){
            int wj = w*j;
            pixPoints1.col((i-1)*3 + (j-1)) = Eigen::Vector3f(hi, wj, 1);
        }
    }
    // std::cout<<"pixPoints1:\n"<<pixPoints1<<std::endl;
    // 生成地图点
    mapPoints.resize(pixPoints1.rows(), pixPoints1.cols());
    for(int i=0; i<pixPoints1.cols(); ++i){
        Eigen::Vector3f p1 = pixPoints1.col(i).cast<float>();
        // 避免点在同一平面上
        mapPoints.col(i) = (cam.K_inv * p1) * (1 + 0.1*i);
    }
    // std::cout<<"mapPoints:\n"<<mapPoints<<std::endl;
    // 生成相机2的像素点
    pixPoints2.resize(pixPoints1.rows(), pixPoints1.cols());
    for(int i=0; i<mapPoints.cols(); ++i){
        const Eigen::Vector3d& mp = mapPoints.col(i).cast<double>();
        Eigen::Vector3d pc = Tbw * mp;
        // std::cout<<"pc: "<<pc.transpose()<<std::endl;
        Eigen::Vector3d pix = cam.K.cast<double>() * pc;
        pix /= pix[2];
        pixPoints2.col(i) = pix.cast<float>(); // .cast<int>();
    }
    // std::cout<<"pixPoints2:\n"<<pixPoints2<<std::endl;
#if USE_INVERSE_DEPTH
    pixPoints3.resize(pixPoints1.rows(), pixPoints1.cols());
    for(int i=0; i<mapPoints.cols(); ++i){
        const Eigen::Vector3d& mp = mapPoints.col(i).cast<double>();
        Eigen::Vector3d pc = Tcw * mp;
        // std::cout<<"pc: "<<pc.transpose()<<std::endl;
        Eigen::Vector3d pix = cam.K.cast<double>() * pc;
        pix /= pix[2];
        pixPoints3.col(i) = pix.cast<float>(); // .cast<int>();
    }
    // std::cout<<"pixPoints2:\n"<<pixPoints2<<std::endl;
#endif
}

int main(int argc, char** argv) {
    int its = 100;
    if(argc>1){
        std::string its2(argv[1]);
        its = std::stoi(its2);
    }
    Eigen::MatrixXf pixPoints1, pixPoints2, pixPoints3;
    Eigen::MatrixXf mapPoints;
    generatePoints(pixPoints1, pixPoints2, mapPoints, pixPoints3);
    CameraModel cam;

    // 优化Tbw先
    // Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1.);
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    const double roll2 = 0. * kDegree2Rad; // 5
    const double pitch2 = -20 * kDegree2Rad; // -25
    const double yaw2 = 10. * kDegree2Rad; // 15
    const Eigen::Quaterniond Qwb2 = Eigen::AngleAxisd(yaw2, Eigen::Vector3d::UnitZ()) * \
                            Eigen::AngleAxisd(pitch2, Eigen::Vector3d::UnitY()) * \ 
                            Eigen::AngleAxisd(roll2, Eigen::Vector3d::UnitX());
    const Eigen::Vector3d twb2(-0.4, -0.15, -1.5); // (-0.5, -0.2, -2.0)
    const Sophus::SE3d Twb2(Qwb2, twb2);
    const Sophus::SE3d Tbw2(Twb2.inverse());

    //Qwb2.toRotationMatrix() = Qwb2.toRotationMatrix() * Qwb2.toRotationMatrix().inverse();
    //std::cout<<"Qwb2 be identity?:\n "<<Qwb2.toRotationMatrix()<<std::endl; // no really
    // std::cout<<"Qwb2 address: "<<&Qwb2<<std::endl;
    // error: taking address of temporary
    // std::cout<<"Qwb2 matrix addr: "<<&(Qwb2.toRotationMatrix())<<std::endl;

    // Step 2: 初始化优化器的节点和边
    VertexPose* T_bw = new VertexPose(Tbw2);
    T_bw->setId(0);
    T_bw->setFixed(false);
    optimizer.addVertex(T_bw);

    VertexPose* T_aw = new VertexPose(kTw);
    T_aw->setId(1);
    T_aw->setFixed(true);
    optimizer.addVertex(T_aw);

#if USE_INVERSE_DEPTH
    VertexPose* T_cw = new VertexPose(Tcw);
    T_cw->setId(2);
    // T_cw 加入的约束可以约束地图点的尺度
    // 如果只有T_aw，T_bw约束，那么尺度是不可观的
    T_cw->setFixed(false);
    optimizer.addVertex(T_cw);
#endif

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0., 0.01);
    
    for(int i=0; i<mapPoints.cols(); ++i){
        double noise = distribution(generator);
#if USE_INVERSE_DEPTH
        VertexInverseDepthPoint* p = new VertexInverseDepthPoint(1.0, pixPoints1.col(i));
        p->setId(i+3);
        p->setFixed(false);
        optimizer.addVertex(p);

        const Eigen::Vector2d obs = pixPoints2.col(i).head(2).cast<double>();

        EdgeInverseDepthPoint *e = new EdgeInverseDepthPoint(obs, &cam);
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(1)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(i+3)));
        e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e);

        const Eigen::Vector2d obs2 = pixPoints3.col(i).head(2).cast<double>();

		EdgeInverseDepthPoint *e2 = new EdgeInverseDepthPoint(obs2, &cam);
		e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(1)));
		e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(i+3)));
		e2->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(2)));
		e2->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e2);
#else
        Eigen::Vector3d vecNoise(noise, noise, noise);
        
        const Eigen::Vector3d pw = mapPoints.col(i).cast<double>() + vecNoise;
        // const Eigen::Vector3d pw = mapPoints.col(i).cast<double>();
        VertexPointXYZ* p = new VertexPointXYZ(pw);
        p->setId(i+2);
        p->setFixed(false);
        optimizer.addVertex(p);

        // 相机b的观测
        const Eigen::Vector2d obs = pixPoints2.col(i).head(2).cast<double>();
        EdgeMono *e = new EdgeMono(obs, &cam);
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(i+2)));
        // 信息矩阵默认为0, 而chi2的计算与信息矩阵相关，所以必须设置信息矩阵
        e->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e);

        // 相机a的观测
        const Eigen::Vector2d obs1 = pixPoints1.col(i).head(2).cast<double>();
        EdgeMono *e1 = new EdgeMono(obs1, &cam);
        e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(1)));
        e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(i+2)));
        e1->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e1);
#endif
    }

	// 添加相对位姿约束以保证尺度一致性
	Sophus::SE3d Tab = Sophus::SE3d() * Twb;
	Tab.translation() += Eigen::Vector3d(0.01, 0.03, 0.1);
	PoseConstraint* e = new PoseConstraint(Tab);
	e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(1)));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
    e->setInformation(Matrix6d::Identity());
    optimizer.addEdge(e);

    // Step 3: 执行优化
    optimizer.setVerbose(true);
    optimizer.initializeOptimization(0);
    optimizer.computeActiveErrors();
    std::cout<<"initial optimizer.chi2(): "<<optimizer.chi2()<<std::endl;
    optimizer.optimize(its);

    VertexPose* Tbw_recovery = static_cast<VertexPose*>(optimizer.vertex(0));
    Sophus::SE3d Twb_final = Tbw_recovery->estimate().T.inverse();
    std::cout<<"\n========\nCompare euler angles and translation after optimizating "<<its<<" times:\n";
    std::cout<<"Twb_true: "<<"["<<Twb.rotationMatrix().eulerAngles(2, 1, 0).transpose()*kRad2Degree<<"], ["<<
            Twb.translation().transpose()<<"]\n";

    std::cout<<"Twb_before opt:"<<"["<<Twb2.rotationMatrix().eulerAngles(2, 1, 0).transpose()*kRad2Degree<<"], ["<<
            Twb2.translation().transpose()<<"]\n";
    
    std::cout<<"Twb_after opt:"<<"["<<Twb_final.rotationMatrix().eulerAngles(2, 1, 0).transpose()*kRad2Degree<<"], ["<<
            Twb_final.translation().transpose()<<"]\n";

#if USE_INVERSE_DEPTH
	Sophus::SE3d Twc2 = static_cast<VertexPose*>(optimizer.vertex(2))->estimate().T.inverse();
	std::cout<<"Twc_true: "<<"["<<Twc.rotationMatrix().eulerAngles(2, 1, 0).transpose()*kRad2Degree<<"], ["<<
            Twc.translation().transpose()<<"]\n";
    
    std::cout<<"Twc_after opt:"<<"["<<Twc2.rotationMatrix().eulerAngles(2, 1, 0).transpose()*kRad2Degree<<"], ["<<
            Twc2.translation().transpose()<<"]\n";
#endif
    
    return 0;
}