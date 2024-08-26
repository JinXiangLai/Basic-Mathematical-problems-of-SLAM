#include <Eigen/Dense>

#include <sophus/se3.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_multi_edge.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

template <typename T = double>
Eigen::Matrix<T, 3, 3> NormalizeRotation(const Eigen::Matrix<T, 3, 3> &R)
{
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return NormalizeRotation(res);
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return NormalizeRotation(res);
    }
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

// Θ很小时，近似参考：https://blog.csdn.net/wang_yq0728/article/details/121894502
// 其中 Jr(ϕ) = Jl(-ϕ)
// Jr(ϕ).inv = Jl(-ϕ).inv
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    // 计算旋转角
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    // 旋转向量的反对称矩阵，即 W=Θ*a
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        // W*W/d2 = a * a.T
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d v){
    return InverseRightJacobianSO3(v[0], v[1], v[2]);
}

struct CameraModel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    float f = 0.008; // 焦距
    float pix_x_meter = 2e-5;
    float pix_y_meter = 2e-5;
    float fx = f / pix_x_meter;
    float fy = f / pix_y_meter;
    int img_width = 1920;
    int img_height = 1080;
    int cx = img_width / 2;
    int cy = img_height / 2;
    Eigen::Matrix3f K;
    //K << fx, cx, 0,
    //	0, fy, cy,
    //	0, 0, 1;
    Eigen::Matrix3f K_inv; // = K.inverse();
    CameraModel(){
        K << fx, cx, 0,
        0, fy, cy,
        0, 0, 1;
        K_inv = K.inverse();
    }
};

// point 类型
class PointXYZ {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointXYZ(){};
    PointXYZ(double _x, double _y, double _z){
        xyz = Eigen::Vector3d(_x, _y, _z);
    }
    PointXYZ(const Eigen::Vector3d& p){
        xyz = p;
    }
    // 	需要定义Update函数
    void Update(const double *up){
        Eigen::Vector3d delta(up[0], up[1], up[2]);
        xyz += delta;
    }
    Eigen::Vector3d xyz;
};
class VertexPointXYZ : public g2o::BaseVertex<3, PointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPointXYZ(){};
    VertexPointXYZ(double _x, double _y, double _z){
        PointXYZ xyz(_x, _y, _z);
        setEstimate(xyz);
    }
    VertexPointXYZ(const Eigen::Vector3d& p){
        PointXYZ xyz = p;
        setEstimate(xyz);
    }

    // 需要重新定义的纯虚函数
    virtual bool read(std::istream &is){return false;}
    virtual bool write(std::ostream &os) const{return false;}

    // 重置函数,设定被优化变量的原始值
    virtual void setToOriginImpl()
    {
    }

    virtual void oplusImpl(const double *update_)
    {
        // https://github.com/RainerKuemmerle/g2o/blob/master/doc/README_IF_IT_WAS_WORKING_AND_IT_DOES_NOT.txt
        // 官方讲解cache
        // 需要在oplusImpl与setEstimate函数中添加
        _estimate.Update(update_);
        updateCache();
    }

};


// pose 类型
class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Pose(){};
    Pose(Sophus::SE3d _T):T(_T){}
    void Update(const double *up){
        Eigen::Vector3d r(up[0], up[1], up[2]);
        Eigen::Vector3d t(up[3], up[4], up[5]);
        // Eigen::Matrix3d delta_r = Eigen::AngleAxisd(r.norm(), r.normalized()).toRotationMatrix();
        Eigen::Matrix3d delta_r = ExpSO3(r[0], r[1], r[2]);

        // 这样赋值是错误的，rotationMatrix()函数是const修饰，返回临时变量
        // T.rotationMatrix() = delta_r * T.rotationMatrix();
        T.setRotationMatrix(delta_r * T.rotationMatrix());
        T.translation() += t;
    }
    Sophus::SE3d T;
};
class VertexPose : public g2o::BaseVertex<6, Pose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){};
    VertexPose(Sophus::SE3d _T){
        Pose T(_T);
        setEstimate(_T);
    }

    virtual bool read(std::istream &is){return false;}
    virtual bool write(std::ostream &os)const {return false;}

    // 重置函数,设定被优化变量的原始值
    virtual void setToOriginImpl()
    {
    }

    virtual void oplusImpl(const double *update_)
    {
        _estimate.Update(update_);
        updateCache();
    }
};

// point->pose类型
class EdgeMono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexPointXYZ>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMono(){};

    EdgeMono(const Eigen::Vector2d _p, CameraModel* _cam) : cam(_cam)
    {
        // 像素值为int类型
        setMeasurement(_p.cast<int>().cast<double>());
    }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    // 计算重投影误差
    void computeError()
    {
        const VertexPose* Tcw = static_cast<VertexPose*>(_vertices[0]);
        const VertexPointXYZ* pw = static_cast<VertexPointXYZ*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        
        pc1 = Tcw->estimate().T * pw->estimate().xyz;
        pc2 = cam->K.cast<double>() * pc1;
        Eigen::Vector2d pix = (pc2 / pc2[2]).head(2);
        _error = pix - obs;
        // std::cout<<"e: "<<_error.transpose()<<std::endl;       
    }

    virtual void linearizeOplus(){
        const VertexPose* Tcw = static_cast<VertexPose*>(_vertices[0]);
        const VertexPointXYZ* _pw = static_cast<VertexPointXYZ*>(_vertices[1]);

        const Eigen::Matrix3d Rbw = Tcw->estimate().T.rotationMatrix();
        const Eigen::Vector3d tbw = Tcw->estimate().T.translation();
        const Eigen::Vector3d pw = _pw->estimate().xyz;

        // residual [r1, r2] about pc2[px, py, pz]
        Eigen::Matrix<double, 2, 3> J_r_pc2;
        J_r_pc2 << 1/pc2[2], 0, -pc2[0]/(pc2[2]*pc2[2]),
                    0, 1/pc2[2], -pc2[1]/(pc2[2]*pc2[2]);
        // pc2 about pc1
        // const double cx = cam->cx, cy = cam->cy, fx = cam->fx, fy = cam->fy;
        Eigen::Matrix3d J_pc2_pc1 = cam->K.cast<double>();
        // J_pc2_pc1<< cx, 0, fx, \
                    0, cy, fy, \
                    0, 0, 1;
        // pc1 about pw
        Eigen::Matrix3d J_pc1_pw = Rbw;
        _jacobianOplusXj = J_r_pc2 * J_pc2_pc1 * J_pc1_pw;

        // Jacobian [r1, r2] about [R, t]
        // pc1 w.r.t. Rbw
        Eigen::Matrix3d J_pc1_Rbw = Sophus::SO3d::hat(-Rbw * pw);
        // pc1 w.r.t tbw
        Eigen::Matrix3d J_pc1_tbw = Eigen::Matrix3d::Identity();
        // pc1 w.r.t. Tbw
        Eigen::Matrix<double, 3, 6> J_pc1_Tbw;
        J_pc1_Tbw.block<3, 3>(0, 0) = J_pc1_Rbw;
        J_pc1_Tbw.block<3, 3>(0, 3) = J_pc1_tbw;
        _jacobianOplusXi = J_r_pc2 * J_pc2_pc1 * J_pc1_Tbw;
    }
    CameraModel* cam{nullptr};

    Eigen::Vector3d pc1, pc2;
    Eigen::Vector3d pix;
};

// 定义详细的节点参数以及Update过程
class InverseDepthPoint{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InverseDepthPoint() {}
    InverseDepthPoint(double _rho, int u, int v) : rho(_rho){
        uv<< u, v, 1;
    }
    InverseDepthPoint(double _rho, const Eigen::Vector3f& _uv) : rho(_rho){
        uv = _uv.cast<double>();
    }
    void Update(const double* up){
        rho += up[0];
    }
    double rho; // 1/Zc
    Eigen::Vector3d uv; // host frame 的像素坐标,非优化变量

};

// 将节点封装成g2o::Vertex类型即可
class VertexInverseDepthPoint : public g2o::BaseVertex<1, InverseDepthPoint>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexInverseDepthPoint() {}
    VertexInverseDepthPoint(double _rho, int u, int v){
        InverseDepthPoint p = InverseDepthPoint(_rho, u, v);
        setEstimate(p);
    }
    VertexInverseDepthPoint(double _rho, const Eigen::Vector3f& uv){
        InverseDepthPoint p = InverseDepthPoint(_rho, uv);
        setEstimate(p);
    }

    virtual bool read(std::istream &is){return false;}
    virtual bool write(std::ostream &os)const {return false;}
    virtual void setToOriginImpl()
    {
    }

    virtual void oplusImpl(const double *update_)
    {
        _estimate.Update(update_);
        updateCache();
    }
};

// 完成观测值设置以及误差计算和雅可比函数
class EdgeInverseDepthPoint : public g2o::BaseMultiEdge<2, Eigen::Vector2d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseDepthPoint() {
        resize(3);
    }
    EdgeInverseDepthPoint(const Eigen::Vector2d& _obs, CameraModel* _cam, const double& _w = 1.0){
        resize(3);
        cam = _cam;
        setMeasurement(_obs);
        setInformation(Eigen::Matrix2d::Identity() * _w);
    }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    // 计算重投影误差
    void computeError()
    {
        const VertexPose* Tfw = static_cast<VertexPose*>(_vertices[0]); // host frame
        const VertexInverseDepthPoint* pf = static_cast<VertexInverseDepthPoint*>(_vertices[1]);
        const VertexPose* Tcw = static_cast<VertexPose*>(_vertices[2]); // current frame

        const Sophus::SE3d T_fw = Tfw->estimate().T;
        const Eigen::Vector2d obs(_measurement);
        const Eigen::Matrix3d K_invd = cam->K_inv.cast<double>();
        const Eigen::Matrix3d Kd = cam->K.cast<double>();        

        pc_f = 1./pf->estimate().rho * K_invd * pf->estimate().uv;
        pw = T_fw.rotationMatrix().transpose() * (pc_f - T_fw.translation());
        pc1 = Tcw->estimate().T * pw;
        pc2 = Kd * pc1;
        pix = (pc2/pc2[2]);

        _error = pix.head(2) - obs;
        // std::cout<<"e: "<<_error.transpose()<<std::endl;       
    }

    virtual void linearizeOplus(){
        const VertexPose* Tfw = static_cast<VertexPose*>(_vertices[0]); // host frame
        const VertexInverseDepthPoint* pf = static_cast<VertexInverseDepthPoint*>(_vertices[1]);
        const VertexPose* Tcw = static_cast<VertexPose*>(_vertices[2]); // current frame

        const Sophus::SE3d T_fw = Tfw->estimate().T.inverse();
        const Eigen::Matrix3d Rfw = T_fw.rotationMatrix();
        const Eigen::Vector3d tfw = T_fw.translation();
        const Eigen::Matrix3d Rcw = Tcw->estimate().T.rotationMatrix();
        const Eigen::Vector3d tcw = Tcw->estimate().T.translation();

        // pix w.r.t pc2
        Eigen::Matrix<double, 2, 3> J_r_pc2;
        J_r_pc2 << 1/pc2[2], 0, -pc2[0]/(pc2[2]*pc2[2]),
                    0, 1/pc2[2], -pc2[1]/(pc2[2]*pc2[2]);
        // pc2 w.r.t pc1
        Eigen::Matrix3d J_pc2_pc1 = cam->K.cast<double>();
        // pc1 w.r.t pw
        Eigen::Matrix3d J_pc1_pw = Rcw;

        for(int i=0; i<3; ++i){
            _jacobianOplus[i].setZero();
        }

        // Jacobian of residual w.r.t pw
        Eigen::Matrix<double, 2, 3> J_r_pw = J_r_pc2 * J_pc2_pc1 * J_pc1_pw;

        // Jacobian of residual w.r.t Tfw
        // pw w.r.t Rfw
        Eigen::Vector3d pf_minus_tfw = pc_f - tfw;
        Eigen::Matrix3d J_pw_Rfw = Rfw.transpose() * Sophus::SO3d::hat(pf_minus_tfw);
        // pw w.r.t tfw
        Eigen::Matrix3d J_pw_tfw = -Rfw.transpose();
        
        Eigen::Matrix<double, 3, 6> J_pw_Tfw;
        J_pw_Tfw.block<3, 3>(0, 0) = J_pw_Rfw;
        J_pw_Tfw.block<3, 3>(0, 3) = J_pw_tfw;
        _jacobianOplus[0] = J_r_pw * J_pw_Tfw;

        // Jacobian of residual w.r.t rho
        // pw w.r.t pc_f
        Eigen::Matrix3d pw_pcf = Rfw.transpose();
        // pc_f w.r.t rho
        Eigen::Vector3d pcf_norm = cam->K_inv.cast<double>() * pf->estimate().uv;
        double rho2 = pf->estimate().rho * pf->estimate().rho;
        Eigen::Vector3d pcf_rho( -pcf_norm[0]/rho2, -pcf_norm[1]/rho2, -1/rho2 );
        _jacobianOplus[1] = J_r_pw * pw_pcf * pcf_rho;

        // Jacobian of residual w.r.t Tcw
        // pc1 w.r.t Rcw
        Eigen::Matrix3d J_pc1_Rcw = -Sophus::SO3d::hat(Rcw * pw);
        Eigen::Matrix3d J_pc1_tcw = Eigen::Matrix3d::Identity();
        
        Eigen::Matrix<double, 3, 6> J_pc1_Tcw;
        J_pc1_Tcw.block<3, 3>(0, 0) = J_pc1_Rcw;
        J_pc1_Tcw.block<3, 3>(0, 3) = J_pc1_tcw;
        _jacobianOplus[2] = J_r_pc2 * J_pc2_pc1 * J_pc1_Tcw;
    }

    Eigen::Vector3d pc_f, pw, pc1, pc2, pix;

    CameraModel* cam{nullptr};
};

class PoseConstraint : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexPose, VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PoseConstraint(){}
    PoseConstraint(const Sophus::SE3d& _Tab){
        setMeasurement(_Tab);
        setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError(){
        const VertexPose* Taw = static_cast<VertexPose*>(_vertices[0]);
        const VertexPose* Tbw = static_cast<VertexPose*>(_vertices[1]);
        const Eigen::Matrix3d& Raw = Taw->estimate().T.rotationMatrix();
        const Eigen::Vector3d& taw = Taw->estimate().T.translation();
        const Eigen::Matrix3d& Rbw = Tbw->estimate().T.rotationMatrix();
        const Eigen::Vector3d& tbw = Tbw->estimate().T.translation();

        const Eigen::Matrix3d& Rab = _measurement.rotationMatrix();
        const Eigen::Vector3d& tab = _measurement.translation();

        delta_R =  LogSO3(Rab.transpose() * Raw * Rbw.transpose());
        Eigen::Vector3d delta_t =  tab + Raw*Rbw.transpose()*tbw - taw;
        _error << delta_R, delta_t;
    
    }

    virtual void linearizeOplus(){
        const VertexPose* Taw = static_cast<VertexPose*>(_vertices[0]);
        const VertexPose* Tbw = static_cast<VertexPose*>(_vertices[1]);
        const Eigen::Matrix3d& Raw = Taw->estimate().T.rotationMatrix();
        const Eigen::Vector3d& taw = Taw->estimate().T.translation();
        const Eigen::Matrix3d& Rbw = Tbw->estimate().T.rotationMatrix();
        const Eigen::Vector3d& tbw = Tbw->estimate().T.translation();

        const Eigen::Matrix3d& Rab = _measurement.rotationMatrix();
        const Eigen::Vector3d& tab = _measurement.translation();

        // inverse left Jacobian of deltaR
        // Jr(ϕ).inv = Jl(-ϕ).inv
        Eigen::Matrix3d invJl = InverseRightJacobianSO3(-delta_R);
        Eigen::Matrix3d invJr = InverseRightJacobianSO3(delta_R);

        // Jacobian delta_R w.r.t Raw(左扰动)
        // 使用伴随矩阵以及BCH近似两个性质
        Eigen::Matrix3d J_deltaR_Raw = invJl * Rab;

        // Jacobian delta_R w.r.t Rbw
        Eigen::Matrix3d J_deltaR_Rbw = -invJr; 
        // Jacobian delta_R w.r.t taw(tbw) is 0

        // Jacobian delta_t w.r.t Raw
        Eigen::Matrix3d J_deltat_Raw = -Sophus::SO3d::hat(Raw * Rbw.transpose() * tbw);

        // Jacobian delta_t w.r.t Rbw
        Eigen::Matrix3d J_deltat_Rbw = Raw * Rbw.transpose() * Sophus::SO3d::hat(tbw);

        // Jacobian deltat w.r.t taw
        Eigen::Matrix3d J_deltat_taw = -Eigen::Matrix3d::Identity();

        // Jacobian deltat w.r.t tbw
        Eigen::Matrix3d J_deltat_tbw = Raw*Rbw.transpose();

        _jacobianOplusXi.setZero();
        _jacobianOplusXj.setZero();

        _jacobianOplusXi.block<3, 3>(0, 0) = J_deltaR_Raw;
        _jacobianOplusXi.block<3, 3>(3, 0) = J_deltat_Raw;
        _jacobianOplusXi.block<3, 3>(3, 3) = J_deltat_taw;

        _jacobianOplusXj.block<3, 3>(0, 0) = J_deltaR_Rbw;
        _jacobianOplusXj.block<3, 3>(3, 0) = J_deltat_Rbw;
        _jacobianOplusXj.block<3, 3>(3, 3) = J_deltat_tbw;

    }
    Eigen::Vector3d delta_R;

};