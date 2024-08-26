import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import sparse
from scipy.linalg import expm, logm
from scipy.sparse.linalg import spsolve
import time
from typing import List


class Sophus():

    def __init__(self, x, y, z, qw, qx, qy, qz):
        self.pos_ = np.array([x, y, z], dtype=float)
        rot = np.array([qw, qx, qy, qz], dtype=float)
        rot_vec = Sophus.quat2rotvec(rot)
        pos_rot_vec = np.array([x, y, z, rot_vec[0], rot_vec[1], rot_vec[2]])
        self.T_ = self.exp(pos_rot_vec)
        self.T_[:3, 3] = self.pos_

    @property
    def pos(self) -> np.ndarray:
        return self.pos_

    @staticmethod
    def quat2rotvec(quat) -> np.ndarray:
        qv = quat[1:]
        qw = quat[0]
        qv_norm = np.linalg.norm(qv)
        if qv_norm < 1e-7:
            print('qv_norm= ', qv_norm, '\n')
            rot_vec = 2 * qv
            return rot_vec
        theta = 2 * np.arctan2(qv_norm, qw)
        theta = Sophus.pi2pi(theta)
        rot_vec = theta * qv / qv_norm
        return rot_vec

    @staticmethod
    def pi2pi(rad):
        val = np.fmod(rad, 2.0 * np.pi)
        if val > np.pi:
            val -= 2.0 * np.pi
        elif val < -np.pi:
            val += 2.0 * np.pi

        return val

    @staticmethod
    def skew_symmetric(rot_vec):
        v = rot_vec
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                        dtype=float)

    '''
        lgom and expm function consume about 300ms on one call, too slow!!!
    '''
    @staticmethod
    def exp(pos_rot_vec: np.ndarray) -> np.ndarray:
        skew_mat = np.zeros((4, 4), dtype=float)
        skew_mat[:3, :3] = Sophus.skew_symmetric(pos_rot_vec[3:])
        skew_mat[:3, 3] = pos_rot_vec[:3]
        SE3 = expm(skew_mat)
        return SE3

    @staticmethod
    def log(SE3_matrix: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/73900095/can-i-discard-the-complex-portion-of-results-generated-with-scipy-linalg-logm
        skew_mat = np.real(logm(SE3_matrix))
        logm(SE3_matrix)
        x, y, z = skew_mat[:3, 3]
        a, b, c = skew_mat[2, 1], skew_mat[0, 2], skew_mat[1, 0]
        return np.array([x, y, z, a, b, c])

    def update(self, pos_rot_vec: np.ndarray) -> None:
        delta_SE3 = Sophus.exp(pos_rot_vec)
        self.T_ = np.matmul(delta_SE3, self.T_)


class Constraint():

    def __init__(self, id0: int, id1: int, t_v1_v2: np.ndarray,
                 info_mat: np.ndarray):
        self.id0_ = id0
        self.id1_ = id1
        x, y, z, qw, qx, qy, qz = t_v1_v2
        t = Sophus(x, y, z, qw, qx, qy, qz)
        self.t_inv_mat_ = np.linalg.inv(t.T_)
        self.info_mat_ = info_mat

    @property
    def id0(self):
        return self.id0_

    @property
    def id1(self):
        return self.id1_

    @property
    def info_mat(self):
        return self.info_mat_

    @staticmethod
    def adjoint_matrix(mat):
        '''
                    | R  t^R |
            Ad(T) = |        |
                    | 0   R  |
        '''
        R = mat[:3, :3]
        t = mat[:3, 3]
        t_skew_sym = Sophus.skew_symmetric(t)
        t_skew_sym_R = t_skew_sym @ R
        ad = np.zeros((6, 6), dtype=float)
        ad[:3, :3] = R
        ad[:3, 3:] = t_skew_sym_R
        ad[3:, 3:] = R
        return ad

    def calculate_error(self, v1: Sophus, v2: Sophus)->np.ndarray:
        '''
            err = T^-1 * T_a^-1 * T_b ==>
            --------------------------------------------------------------
            T_a^-1 ==>
            | R_a^-1  |  -R_a^-1 * t_a |
            |    0    |         1      |
            --------------------------------------------------------------
            | R_a^-1  | -R_a^-1 * t_a |   | R_b  |  t_b |
            |   0     |      1        | * |  0   |   1  | ===>
            --------------------------------------------------------------
            | R_a^-1 * R_b  |  R_a^-1 * t_b - R_a^-1 * t_a |
            |       0       |               1              |
            --------------------------------------------------------------
        '''
        v_1_inv_mat = np.linalg.inv(v1.T_)
        e_mat = self.t_inv_mat_ @ v_1_inv_mat @ v2.T_
        e_vec = Sophus.log(e_mat)

        return e_vec

    def calculate_jacobian(self, v1: Sophus, v2: Sophus):
        '''
            ∂e_ij
           —————— = {-J_r^-1(e_ij)} * Ad(T_j^-1)
            ∂δξ_i
            ------------------------------------
            ∂e_ij
           —————— = {J_r^-1(e_ij)} * Ad(T_j^-1)
            ∂δξ_j
            =====================================
                                     | Φ_e^  ρ_e^ |
            J_r^-1(e_ij) = I + 1/2 * |            |
                                     | 0     Φ_e^ |
        '''
        e = self.calculate_error(v1, v2)
        e_rou = e[:3]  # not equal to position in SE3
        e_rotvec = e[3:]

        v_2_inv_mat = np.linalg.inv(v2.T_)
        ad_v2_inv = Constraint.adjoint_matrix(v_2_inv_mat)

        Jr_inv = np.zeros((6, 6), dtype=float)
        Jr_inv[:3, :3] = Sophus.skew_symmetric(e_rotvec)
        Jr_inv[:3, 3:] = Sophus.skew_symmetric(e_rou)
        Jr_inv[3:, 3:] = Jr_inv[:3, :3]
        Jr_inv = np.identity(6, dtype=float) + 0.5 * Jr_inv

        J1 = -1 * Jr_inv @ ad_v2_inv
        J2 = Jr_inv @ ad_v2_inv

        return e, J1, J2


class PgoOptimizer():

    def __init__(self, iteration, converge, g2o_file_path):
        self.file_path_ = g2o_file_path
        self.iteration_ = iteration
        self.converge_ = converge
        self.vertexs_ = []
        self.constraints_ = []
        self.robust_delta = 1.0

    def robust_coeff(self, squared_error, delta):
        if squared_error < 0:
            return 0
        sqre = np.sqrt(squared_error)
        if sqre < delta:
            return 1  # no effect
        return delta / sqre  # linear

    def load_g2o_file(self):
        vertexs, constraints = [], []

        for line in open(self.file_path_):
            sline = line.split()
            tag = sline[0]

            if tag == "VERTEX_SE3:QUAT":
                # data_id = int(sline[1]) # unused
                x = float(sline[2])
                y = float(sline[3])
                z = float(sline[4])
                qx = float(sline[5])
                qy = float(sline[6])
                qz = float(sline[7])
                qw = float(sline[8])

                vertexs.append(Sophus(x, y, z, qw, qx, qy, qz))
            elif tag == "EDGE_SE3:QUAT":
                id1 = int(sline[1])
                id2 = int(sline[2])
                x = float(sline[3])
                y = float(sline[4])
                z = float(sline[5])
                qx = float(sline[6])
                qy = float(sline[7])
                qz = float(sline[8])
                qw = float(sline[9])
                c1 = float(sline[10])
                c2 = float(sline[11])
                c3 = float(sline[12])
                c4 = float(sline[13])
                c5 = float(sline[14])
                c6 = float(sline[15])
                c7 = float(sline[16])
                c8 = float(sline[17])
                c9 = float(sline[18])
                c10 = float(sline[19])
                c11 = float(sline[20])
                c12 = float(sline[21])
                c13 = float(sline[22])
                c14 = float(sline[23])
                c15 = float(sline[24])
                c16 = float(sline[25])
                c17 = float(sline[26])
                c18 = float(sline[27])
                c19 = float(sline[28])
                c20 = float(sline[29])
                c21 = float(sline[30])
                t = np.array([x, y, z, qw, qx, qy, qz], dtype=float)
                info_mat = np.array(
                    [[c1, c2, c3, c4, c5, c6], [c2, c7, c8, c9, c10, c11],
                     [c3, c8, c12, c13, c14, c15], [
                         c4, c9, c13, c16, c17, c18
                     ], [c5, c10, c14, c17, c19, c20],
                     [c6, c11, c15, c18, c20, c21]],
                    dtype=float)
                constraints.append(Constraint(id1, id2, t, info_mat))  # 添加约束

        self.vertexs_ = vertexs
        self.constraints_ = constraints

        print("n_vertexs:", len(vertexs))
        print("n_constraints:", len(constraints))

    def calculate_H_b(self):
        '''
                            optimized variables
                        -----------------------------
                        x1  y1  z1  a1  b1  c1      x2......
                   |Δx (0,0)                        (0,6*id)
                   |Δy
            errors |Δz
                   |Δa
                   |Δb
                   |Δc                  (0+6, 0+6)
                   ... (6*id,id)
        '''
        vertesx_num = len(self.vertexs_)
        constraint_num = len(self.constraints_)
        H_col_num = 6 * vertesx_num
        H_row_num = 6 * constraint_num
        J = sparse.lil_matrix((H_row_num, H_col_num), dtype=float)
        f = np.zeros(H_row_num, dtype=float)
        Info = sparse.lil_matrix((H_row_num, H_row_num), dtype=float)

        for i, constraint in enumerate(self.constraints_):
            id0 = constraint.id0
            id1 = constraint.id1
            v0 = self.vertexs_[id0]
            v1 = self.vertexs_[id1]
            e, J1, J2 = constraint.calculate_jacobian(v0, v1)

            Info_matrix = constraint.info_mat * self.robust_coeff(
                e.reshape(6, 1).T @ constraint.info_mat @ e.reshape(6, 1),
                self.robust_delta)
            f[i * 6:i * 6 + 6] = e
            J[i * 6:i * 6 + 6, 6 * id0:6 * id0 + 6] = J1
            J[i * 6:i * 6 + 6, 6 * id1:6 * id1 + 6] = J2
            Info[i * 6:i * 6 + 6, i * 6:i * 6 + 6] = Info_matrix

        H = J.T @ Info @ J
        b = -1 * J.T @ Info @ f.reshape(-1, 1)
        cost = 0.
        for i in range(0, len(self.vertexs_), 6):
            cost += f[i*6:i*6+6].reshape(1,-1) @ Info[i:i+6, i:i+6] @ f[i*6:i*6+6].reshape(-1,1)
        print('current cost = ', cost)
        return H, b, cost

    def solve_H_deltax_b(self) -> np.ndarray:
        H, b, cost = self.calculate_H_b()
        # I = sparse.identity(H_sparse.shape[0], dtype='float')* 1e-7
        # H = I + H_sparse
        for i in range(6):
            H[i, i] = 1.e10  # fix the first pose
        delta_x = spsolve(H, b)
        delta_x = delta_x.reshape(-1)
        return delta_x, cost

    def run(self):
        last_cost = 1.e300
        for it in range(self.iteration_):
            st = time.time()
            delta_x, cost = self.solve_H_deltax_b()
            for i in range(len(self.vertexs_)):
                pos_rotvec = delta_x[i * 6:i * 6 + 6]
                self.vertexs_[i].update(pos_rotvec)
            et = time.time()
            print('iteration %d spend time %f s\n============' %(it, et - st))
            if np.abs(last_cost-cost/len(self.constraints_)) < self.converge_:
                print('Run succeed!\n')
                break


def draw_result(vertexs: List):
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    z0 = []
    z1 = []
    for v in vertexs:
        x0.append(v.pos[0])
        y0.append(v.pos[1])
        z0.append(v.pos[2])
        x1.append(v.T_[:3, 3][0])
        y1.append(v.T_[:3, 3][1])
        z1.append(v.T_[:3, 3][2])
    x0 = np.array(x0)
    y0 = np.array(y0)
    z0 = np.array(z0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    z1 = np.array(z1)

    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x0, y0, z0, c='b', label='before')
    ax.plot(x1, y1, z1, c='r', label='after')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


def main():
    optimizer = PgoOptimizer(30, 0.001, 'sphere2200.g2o')
    optimizer.load_g2o_file()
    optimizer.run()
    draw_result(optimizer.vertexs_)


if __name__ == '__main__':
    main()