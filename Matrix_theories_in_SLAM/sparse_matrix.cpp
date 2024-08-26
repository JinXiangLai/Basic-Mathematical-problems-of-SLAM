#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

struct Triplet{
    Triplet(const int i, const int j, const double value):row(i), col(j), val(value){}
    Triplet(): row(0), col(0), val(0){}
    int row;
    int col;
    double val;
};

class SparseMatrix{
public:
    SparseMatrix(const int row_num, const int col_num, const vector<Triplet>& data){
        row_num_ = row_num;
        col_num_ = col_num;
        for(const Triplet& d : data){
            row_col_data_[d.row][d.col] = d.val;
            col_row_data_[d.col][d.row] = d.val;
        }
    }

    SparseMatrix(){
        row_num_ = 0;
        col_num_ = 0;
    }

    SparseMatrix operator * (const SparseMatrix rhs){
        if(col_num_!=rhs.row_num_){
            return SparseMatrix();
        }
        // 初始化返回值
        SparseMatrix res(row_num_, rhs.col_num_, {});

        // res[i][j] = mat1的第i行乘以mat2第j列的求和
        // 遍历mat1的每一行
        for(const auto& row_data : row_col_data_){
            // mat1第i行的数据
            const int i = row_data.first;
            const unordered_map<int, double>& row1 = row_data.second;

            // 遍历mat2的每一列
            for(const auto& col_data : rhs.col_row_data_){
                const int j = col_data.first;
                const unordered_map<int, double>& col2 = col_data.second;
                
                // 拿到了mat1的第i行和mat2的第j列
                // 遍历第i行的元素来计算 res[i][j]
                for(const auto& d1 : row1){
                    // 第idx列乘以第idx行
                    const int idx = d1.first;
                    if(col2.count(idx)){
                        // 更新返回矩阵的内容
                        const double product = d1.second * col2.at(idx); // 使用 const时不能使用col2[0];
                        res.row_col_data_[i][j] += product;
                        res.col_row_data_[j][i] += product;
                    }
                }
            }
        }

        return res;
    }

    // 成员函数，隐含this指针
    ostream& operator<< ( const ostream& out){
        for(int i=0; i<row_num_; ++i){
            for(int j=0; j<col_num_; ++j){
                if(row_col_data_.count(i)&&row_col_data_.at(i).count(j)){
                    cout<<row_col_data_[i][j]<<" ";
                }else{
                    cout<<0<<" ";
                }
            }
            cout<<endl;
        }
        return cout;
    }

    // 这样就没有this指针了,但是运算符不能是静态成员函数
    // static ostream& operator<< ( const ostream& out, const SparseMatrix& mat){
    // 声明为友元
    friend ostream& operator<< ( const ostream& out, const SparseMatrix& mat){
        const int row_num = mat.row_num_;
        const int col_num = mat.col_num_;
        for(int i=0; i<row_num; ++i){
            for(int j=0; j<col_num; ++j){
                if(mat.row_col_data_.count(i)&&mat.row_col_data_.at(i).count(j)){
                    cout<<mat.row_col_data_.at(i).at(j)<<" ";
                }else{
                    cout<<0<<" ";
                }
            }
            cout<<endl;
        }
    }
    

private:
    int row_num_;
    int col_num_;
    unordered_map<int, unordered_map<int, double>> row_col_data_;
    unordered_map<int, unordered_map<int, double>> col_row_data_;
};

int main(){
    const int i1 = 3;
    const int j1 = 4;
    Triplet d1(0, 2, 5.);
    Triplet d2(1, 1, 3.);
    Triplet d3(2, 3, 2.);
    Triplet d4(0, 0, 7.);
    Triplet d5(1, 2, 8.);
    Triplet d6(2, 0, 3.);
    SparseMatrix mat1(i1, j1, {d1, d2, d3, d4, d5, d6});
    const int i2 = 4;
    const int j2 = 5;
    Triplet dd1(3, 2, 6.);
    Triplet dd2(0, 3, 9.);
    SparseMatrix mat2 (i2, j2, {d1, d2, d3, d4, d5, d6, dd1, dd2});

    SparseMatrix res = mat1 * mat2;
    // mat1.operator<<(cout);
    cout<<mat1;
    cout<<endl;
    mat2.operator<<(cout);
    cout<<endl;
    res.operator<<(cout);
    cout<<endl;

    return 0;
}