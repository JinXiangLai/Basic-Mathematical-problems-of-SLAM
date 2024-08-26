import numpy as np
import math

cos = math.cos
sin = math.sin
atan2 = math.atan2
Degree2Rad = math.pi / 180
Rad2Degree = 180 / math.pi

# 欧拉角与旋转矩阵转换：https://zhuanlan.zhihu.com/p/45404840
def R_x(ang: float, isDegree: bool = True) -> np.ndarray:
    if (isDegree):
        ang *= Degree2Rad
    R = np.array([
        [1, 0, 0], 
        [0, cos(ang), -sin(ang)], 
        [0, sin(ang), cos(ang)]
    ])
    return R

def R_y(ang: float, isDegree: bool = True) -> np.ndarray:
    if (isDegree):
        ang *= Degree2Rad
    R = np.array([
        [cos(ang), 0, sin(ang)], 
        [0, 1, 0], 
        [-sin(ang), 0, cos(ang)]
    ])
    return R

def R_z(ang: float, isDegree: bool = True) -> np.ndarray:
    if (isDegree):
        ang *= Degree2Rad
    R = np.array([
        [cos(ang), -sin(ang), 0], 
        [sin(ang), cos(ang), 0], 
        [0, 0, 1]
    ])
    return R

def R2rpy(R:np.ndarray, seq:str = 'zyx') -> np.ndarray:
    roll = math.atan2(R[2][1], R[2][2])
    if abs(abs(roll)-math.pi) < 1e-3:
        roll = 0.
    pitch = -math.asin(R[2][0])
    if abs(abs(pitch)-math.pi) < 1e-3:
        roll = 0.
    yaw = math.atan2(R[1][0], R[0][0])
    if abs(abs(yaw)-math.pi) < 1e-3:
        yaw = 0.
    return np.array([roll, pitch, yaw]) * Rad2Degree

def skew_symmetric(t) -> np.ndarray:
    a = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    return a

def skew2vec(mat) -> np.ndarray:
    return np.array([mat[2][1], mat[0][2], mat[1][0] ])

# T12
roll_12 = 5
pitch_12 = -25.
yaw_12 = 15
tx_12 = -0.5
ty_12 = -0.2
tz_12 = -2.
R12 = R_z(yaw_12).dot(R_y(pitch_12)).dot(R_x(roll_12))
t_12 = np.array([tx_12, ty_12, tz_12])
T12 = np.eye(4)
T12[:3, :3] = R12
T12[:3, 3] = t_12
#print('T12:\n', T12, '\n============')

T21 = np.linalg.inv(T12)
R21 = T21[:3, :3]
t_21 = T21[:3, 3] 

# 相机内参的逆将像素坐标变换到相机坐标系
# s[u, v, 1] = K * [Xc, Yc, Zc]
f = 0.008 # 焦距
pix_x_meter = 2e-5 
pix_y_meter = 2e-5
fx = f/pix_x_meter
fy = f/pix_y_meter
img_width = 1920
img_height = 1080
cx = img_width/2
cy = img_height/2
K = np.array([
    [fx, cx, 0],
    [0, fy, cy],
    [0, 0, 1]
])
K_inv = np.linalg.inv(K)

# 选取相机1的n个像素点，利用8点法计算本质矩阵E
h = img_height/4
h2 = h * 2
h3 = h * 3
w = img_width/4
w2 = w * 2
w3 = w * 3
points1 = np.array([
    [h, w, 1], [h, w2, 1], [h, w3, 1],
    [h2, w, 1], [h2, w2, 1], [h2, w3, 1],
    [h3, w, 1], [h3, w2, 1], [h3, w3, 1],
    [100, 300, 1], [700, 800, 1], [333, 400, 1],
    [50, 60, 1], [1500, 700, 1], [1555, 777, 1]
])

# 像素点投影到归一化相机坐标系
Points_c1 = points1.copy()
for i in range(points1.shape[0]):
    Points_c1[i] = K_inv.dot(points1[i])

'''
避免所有点共面, 点越不共面，本质矩阵E解算精度越高，这也是
ORB-SLAM要同时解算单应矩阵H的原因
'''
pp = Points_c1.copy()
for i in range(pp.shape[0]):
    pp[i] *= (1 + 0.1 * i)

# 投影到第二个相机上
Points_c2 = pp.copy()
for i in range(Points_c1.shape[0]):
    Points_c2[i] = R21.dot(Points_c2[i]) + t_21
    # pp[i] = R12.dot(Points_c2[i]) + t_12 # 反求验证
print('Points_c2-z: ', Points_c2[:, 2]) # 需要Z值为正

# 投影到第二个相机的像素坐标系
points2 = Points_c2.copy()
for i in range(Points_c2.shape[0]):
    points2[i] = K.dot(Points_c2[i])
    # 理论上应该要取整数
    points2[i] /= points2[i][2]
    # Points_c2也变换到归一化坐标系下
    Points_c2[i] /= Points_c2[i][2]

# 本质矩阵推导
'''
    p2 = R * p1 + t
    t^ * p2 = t^ * R * p1 + t^ * t(=0)
    p2.T * t^ * p2(=0) = p2.T * t^ * R * p1
    ∴ p2.T * t^ * R * p1 = 0
    E = t^ * R
    e = [e0, e1, e2, e3, e4, e5, e6, e7, e8].T
    则 p2.T * t^R * p1 = 0 <==>
    [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1 ] * e(1X9 * 9X1 = 1X1矩阵) = 0,
    构建8对点可求解各个e[i] == > Ax = 0，参考：https://blog.csdn.net/rs_lys/article/details/117736099
    即,限制x模长为1, 对 A.T * A 进行特征值分解，得到的最小特征值对应的特征向量就是特解(最小二乘解)
    
    求解Ax = 0, 参考：https://zhuanlan.zhihu.com/p/447771783
    A = U*Σ*Vh, y = Vh*x
    需要|U*(Σ*Vh*x)|^2取最小值,
    ∵ k = Σ*y是一个vector, U是标准正交矩阵，U*k也是一个向量,
    |U*k|^2 = (U*k).T*(U*k) = k.T * U.T*U(=I) *k
'''
E = skew_symmetric(t_21).dot(R21)

A = np.zeros((points1.shape[0], 9), dtype=float)
for i in range(points1.shape[0]):
    #if(i==8):
    #	break
    p1 = Points_c1[i]
    u1, v1 = p1[0], p1[1]
    p2 = Points_c2[i]
    u2, v2 = p2[0], p2[1]
    A[i] = np.array([
        u2*u1, u2*v1, u2, 
        v2*u1, v2*v1, v2,
        u1, v1, 1
    ])
    # print('should be zero: ', p2.dot(E).dot(p1.transpose()))

print('\n============\nSVD分解ATA求E:')
ATA = A.transpose().dot(A)
# 对ATA进行SVD分解，求出E矩阵
# 理论上来说，A和ATA分解结果应该都是一样的,

sw, v = np.linalg.eig(ATA) # 特征值分解
u, s, vh = np.linalg.svd(A, False)
# e = vh[8] # SVD分解结果
e = v[:, 7] # 特征值分解结果

print('sw: ', sw)
print('s: ', s)
print('E:', E.reshape(1, -1))
print('e: ', e, '\n============')

# 对本质矩阵进行分解
e = e.reshape(3, 3)
print('\n============\nSVD分解E求R, t:')
w1 = R_z(90)
w2 = R_z(-90)
u, s, vh = np.linalg.svd(e)
ue, se, vhe = np.linalg.svd(E)
print('s: ', s)
print('se: ', se)
S = np.diag(s)
#S[1][1] = S[0][0]
S[2][2] = 0.

t1_skew =u.dot(w1).dot(S).dot(u.transpose())
t1 = skew2vec(t1_skew)
print('t_21: ', t_21)
print('t1: ', t1)
print('t_21/t1: ', t_21/t1)

t2_skew = u.dot(w2).dot(S).dot(u.transpose())
t2 = skew2vec(t2_skew)
print('t2: ', t2)
print('t/t2: ', t_21/t2)

print('R21-rpy: ', R2rpy(R21))
R1 = u.dot(w1.transpose()).dot(vh)
# 旋转矩阵的行列式应该为1
det_R1 = np.linalg.det(R1)
print('det(R1): ', det_R1)
if(det_R1 < 0.):
    R1 = -R1
print('R1-rpy: ', R2rpy(R1))
# rpy2R([-1.79999972e+02 -5.06602504e-05 -1.79999990e+02]) = I
rpy_diff1 = R2rpy(R1.transpose() @ R21)
print('R1.inv * R21-rpy: ', rpy_diff1)
print(R1.transpose() @ R21)


R2 = u.dot(w2.transpose()).dot(vh)
det_R2 = np.linalg.det(R2)
print('det(R2): ', det_R2)
if(det_R2 < 0.):
    R2 = -R2
print('R2-rpy: ', R2rpy(R2))
rpy_diff2 = R2rpy(R2.transpose() @ R21)
print('R2.inv * R21-rpy: ', rpy_diff2)
print(R2.transpose() @ R21)

'''
根据重投影找出解参考： https://www.cnblogs.com/houkai/p/6665506.html 
∵ 是根据归一化平面 p2.T * t^R * p1 = 0计算出来的R, t
∴ 将p1通过R、t投影到p2，需要得到正深度才是正确的

使用三维重建恢复地图点深度参考： https://zhuanlan.zhihu.com/p/570001105
========================================
对于归一化坐标系：
s2*p2 = R21 * s1*p1 + t21
p2^ * s2 * p2 = p2^ * R21 * s1*p1 + p2^ * t21 = 0
s1 = -[p2^ * t21] / [p2^ * R21 * p1]
s2 = [R21 * s1*p1 +t21] / p2
'''
print('\n============\n根据多视图几何找正确的R, t:')
Rts = [(R1, t1), (R2, t2), (R1, -t1), (R2, -t2)]
Rts_inv = []
for R, t in Rts:
    Rts_inv.append((R.transpose(), -R.transpose() @ t))

negative_point_num = np.array([0, 0, 0, 0])
for i in range(len(Rts)):
    R, t = Rts[i]
    # 根据投影点数量验证
    for j in range(pp.shape[0]):
    #=================================
    #    p1 = pp[j] 
    #    p2 = R @ p1 + t
    #    if p2[2]<0.:
    #        print('i: ', i, ' j: ', j, '  p2: ', p2)
    #        negative_point_num[i] += 1
    #==================================
        p1 = Points_c1[j]
        p2 = Points_c2[j]
        s1 = -(skew_symmetric(p2) @ t) / (skew_symmetric(p2) @ R21 @ p1)
        s2 = (R.dot(s1[0] * p1) + t)[2]
        # print('s1: ', s1)
        # print('s2: ', s2)
        if s1[0] < 0. or s2 < 0:
            negative_point_num[i] += 1

min_value = negative_point_num[0]
res_id = 0
for i in range(1, len(negative_point_num)):
    if(negative_point_num[i]<min_value):
        res_id = i
        min_value = negative_point_num[i]

# 使用ATA与A进行SVD分解时，得到的R1、R2是类似的，
# 但使用ATA进行特征值分解最终却是得出来错误的结果
print('negative_point_num: ', negative_point_num)
print('result index and min value: ', res_id, ' ', min_value)
print('result R-rpy :', R2rpy(Rts[res_id][0]))
print('true rpy: ', R2rpy(R21))
print('result t: ', Rts[res_id][1])
print('true t: ', t_21)
print('true t/result t: ', t_21/Rts[res_id][1])

'''
使用SVD分解三角化地图点，
设世界点P, 投影矩阵T1[I|0], 投影矩阵T2[R21|t21]
对于归一化图像坐标系(与像素坐标系仅差K)
维度：3X1 = 3X4 * 4X1
s1*p1 = [R1 | t1] * P
s1 * p1^ * p1 = p1^ * [R1 | t1] * P = 0 
同样有
p2^ * [R2 | t2] * P = 0
'''
print('\n============\n使用SVD分解三角化地图点')
T1 = np.zeros((3, 4), float)
T1[:, :3] = np.eye(3)
T2 = np.zeros((3, 4), float)
T2[:, :3] = R21
T2[:, 3] = t_21
for i in range(pp.shape[0]):
    p1 = Points_c1[i]
    p2 = Points_c2[i]
    #A = np.zeros((6, 4), float)
    #A[:3, :] = skew_symmetric(p1) @ T1
    #A[3:, :] = skew_symmetric(p2) @ T2
    # 实际每个观测只有两个线性无关约束
    A = np.zeros((4, 4), float)
    A[:2, :] = (skew_symmetric(p1) @ T1)[:2, :]
    A[2:, :] = (skew_symmetric(p2) @ T2)[:2, :]
    u, s, vh = np.linalg.svd(A)
    p = vh[-1]
    p /= p[3]
    p = p[:3]
    print('p-pp: ', p-pp[i])

# 验证 SVD 分解最小特征值对应的特征向量是 (V.T).T[-1]    
#U = R21
#V = R12
#s = np.diag([3, 2, 1])
#A = U @ s @ V
#AT = U @ s @ V.transpose()
#print('V:\n', R12)
#print('U:\n', U)
#u1, s1, v1 = np.linalg.svd(A)
#print('s1: ', s1)
#print('(V.T).T:\n', V.transpose())
#print('v1:\n', v1)
#print('u1:\n', u1)

#u1, s1, v1 = np.linalg.svd(AT)
#print('s1: ', s1)
#print('(V.T).T:\n', V)
#print('v1:\n', v1)
#print('u1:\n', u1)
