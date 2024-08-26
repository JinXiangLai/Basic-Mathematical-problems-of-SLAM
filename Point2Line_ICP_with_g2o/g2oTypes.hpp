#include <Eigen/Dense>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>

Eigen::Matrix3d hat(const Eigen::Vector3d& omega){
	Eigen::Matrix3d Omega;
	Omega << 0., -omega(2),  omega(1),
             omega(2), 0., -omega(0),
             -omega(1),  omega(0), 0.;
	return Omega;
}

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

class EigenPose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EigenPose(){};
    EigenPose(const Eigen::Quaterniond& q, const Eigen::Vector3d t): qrb(q), trb(t){}
	EigenPose(const Eigen::Matrix3d& R, const Eigen::Vector3d t): trb(t){
		qrb = Eigen::Quaterniond(R);
		qrb.normalize();
	}
    void Update(const double *up){
        Eigen::Vector3d r(up[0], up[1], up[2]);
        Eigen::Vector3d t(up[3], up[4], up[5]);
        // Eigen::Matrix3d delta_r = Eigen::AngleAxisd(r.norm(), r.normalized()).toRotationMatrix();
        Eigen::Matrix3d delta_r = ExpSO3(r[0], r[1], r[2]);
		Eigen::Quaterniond delta_q(delta_r);
		delta_q.normalize();
		qrb = qrb * delta_q;
		trb += t;
    }
	// body frame to reference frame
    Eigen::Quaterniond qrb;
	Eigen::Vector3d trb;
};

class VertexEigenPose : public g2o::BaseVertex<6, EigenPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexEigenPose(){};
    VertexEigenPose(const Eigen::Quaterniond& q, const Eigen::Vector3d t){
        EigenPose Trb(q, t);
        setEstimate(Trb);
    }
	VertexEigenPose(const Eigen::Matrix3d& R, const Eigen::Vector3d t){
		EigenPose Trb(R, t);
		setEstimate(Trb);
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

class EdgePoint2LineICP : public g2o::BaseUnaryEdge <3, Eigen::Vector3d, VertexEigenPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePoint2LineICP(){};

    EdgePoint2LineICP(const Eigen::Vector3d& _ref_p1, const Eigen::Vector3d& _ref_p2, const Eigen::Vector3d& point)
	: ref_p1(_ref_p1)
	, ref_p2(_ref_p2)
	, p(point){}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        // a----->b
		//   \     
        //    c
		// 直线的方向向量
		ref_line = ref_p2 - ref_p1;
		ref_line.normalize();

		// 点变换到参考坐标系
		const VertexEigenPose* Trb = static_cast<VertexEigenPose*>(_vertices[0]);
		const Eigen::Vector3d pr = Trb->estimate().qrb * p + Trb->estimate().trb;

		// 待优化点的向量
		const Eigen::Vector3d line = pr - ref_p1;

		// 根据向量运算计算点到直线距离的向量，为0时残差最小
		// a * b = |a| * |b| * cos(θ)
		_error = line - (line.dot(ref_line)) * ref_line;     
    }

	// 分两步：先求残差关于(Pw_x, Pw_y, Pw_z)的导数，再求Pw关于Twb的导数
    virtual void linearizeOplus(){
		// 注：向量均要看成列向量的形式
		// 记: pr = (xr, yr, zr),
		// ref_line = (x0, y0, z0),
		// ref_p1 = (x1, y1, z1)
		// 则： line = (xr-x1, yr-y1, zr-z1)
		// 残差为： e = (xr-x1, yr-y1, zr-z1) - 
		// [(xr-x1)*x0 + (yr-y1)*y0 + (zr-z1)*z0] * (x0, y0, z0)
		// 整合一下: e = 
		// [ (xr-x1) - (xr-x1)*x0*x0 - (yr-y1)*y0*x0 - (zr-z1)*z0*x0,
		//   (yr-y1) - (xr-x1)*x0*y0 - (yr-y1)*y0*y0 - (zr-z1)*z0*y0,
		//   (zr-z1) - (xr-x1)*x0*z0 - (yr-y1)*y0*z0 - (zr-z1)*z0*z0 ]
		// 可以发现，残差e虽然很长，但都是一次项，容易求导

		// const double xr = pr[0], yr = pr[1], zr = pr[2];
		const double x0 = ref_line[0], y0 = ref_line[1], z0 = ref_line[2];
		// const double x1 = ref_p1[0], y1 = ref_p1[1], z1 = ref_p1[2];

		Eigen::Matrix<double, 3, 3> jac_e_pr = Eigen::Matrix<double, 3, 3>::Zero();
		// jacobian of e w.r.t xr, yr, zr
		jac_e_pr.row(0) << (1-x0*x0), -y0*x0, -z0*x0;
		jac_e_pr.row(1) << -x0*y0, (1-y0*y0), -z0*y0;
		jac_e_pr.row(2) << -x0*z0, -y0*z0, (1-z0*z0);

		const VertexEigenPose* Trb = static_cast<VertexEigenPose*>(_vertices[0]);
		const Eigen::Matrix3d Rrb = Trb->estimate().qrb.toRotationMatrix();
		Eigen::Matrix<double, 3, 6> jac_pr_Rt = Eigen::Matrix<double, 3, 6>::Zero();
		// jacobian of pr w.r.t Rrb, t_rb
		// Pr = Rrb * Pb + t_rb
		jac_pr_Rt.block<3, 3>(0, 0) = -Rrb * hat(p);
		jac_pr_Rt.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
		_jacobianOplusXi = jac_e_pr * jac_pr_Rt;
		
    }

	Eigen::Matrix<double, 6, 6> GetHessian(){
		computeError();
		double dataB[18] = {0.};
		new (&_jacobianOplusXi) JacobianXiOplusType(dataB, 3, 6);
		linearizeOplus();
		// std::cout << _jacobianOplusXi.transpose() * _information * _jacobianOplusXi << std::endl;
		return _jacobianOplusXi.transpose() * _information * _jacobianOplusXi;
	}

	Eigen::Matrix<double, 3, 6> GetJacobian(){
		computeError();
		double dataB[18] = {0.};
		new (&_jacobianOplusXi) JacobianXiOplusType(dataB, 3, 6);
		linearizeOplus();
		return _jacobianOplusXi;
	}

    Eigen::Vector3d ref_p1;
	Eigen::Vector3d ref_p2;
	Eigen::Vector3d p;

	//===== 求雅可比要用到的哦 ====//
private:
	Eigen::Vector3d ref_line = Eigen::Vector3d::Zero();
	// Eigen::Vector3d pr = Eigen::Vector3d::Zero();
	// Eigen::Vector3d line = Eigen::Vector3d::Zero();
};