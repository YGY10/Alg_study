#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "../eigen-3.4.0/Eigen/Dense"
#include "../plot/matplotlibcpp.h"

namespace plt = matplotlibcpp;

// 卡尔曼滤波器类（1D）
class KalmanFilter {
   public:
    KalmanFilter(int state_dim, int measure_dim, int control_dim)
        : x(state_dim),
          P(state_dim, state_dim),
          A_(state_dim, state_dim),
          B_(state_dim, control_dim),
          H_(measure_dim, state_dim),
          Q_(state_dim, state_dim),
          R_(measure_dim, measure_dim),
          I(Eigen::MatrixXd::Identity(state_dim, state_dim)) {}

    // 初始化
    void init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& p0) {
        x = x0;
        P = p0;
    }

    // 设置模型参数
    void setModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H,
                  const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
        A_ = A;
        B_ = B;
        H_ = H;
        Q_ = Q;
        R_ = R;
    }

    // 预测
    void predict(const Eigen::VectorXd& u) {
        x = A_ * x + B_ * u;
        P = A_ * P * A_.transpose() + Q_;
    }

    Eigen::VectorXd getState() const { return x; }

    // 更新
    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd y = z - H_ * x;
        Eigen::MatrixXd S = H_ * P * H_.transpose() + R_;
        Eigen::MatrixXd K = P * H_.transpose() * S.inverse();
        x = x + K * y;
        P = (I - K * H_) * P;
    }

   private:
    Eigen::VectorXd x;   // 状态向量
    Eigen::MatrixXd P;   // 协方差
    Eigen::MatrixXd A_;  // 状态转移
    Eigen::MatrixXd B_;  // 控制矩阵
    Eigen::MatrixXd H_;  // 观测矩阵
    Eigen::MatrixXd Q_;  // 过程噪声
    Eigen::MatrixXd R_;  // 观测噪声
    Eigen::MatrixXd I;   // 单位矩阵
};

int main() {
    int n = 2;  // 状态维度：位置 + 速度
    int m = 1;  // 观测维度：位置
    int l = 1;  // 控制输入维度：加速度

    KalmanFilter kf(n, m, l);

    // 初始状态
    Eigen::VectorXd x0(n);
    x0 << 0.0, 0.0;  // 默认初始位置和速度都默认为0
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 0.01;
    kf.init(x0, P0);

    double dt = 1.0;

    // 模型矩阵
    Eigen::MatrixXd A(n, n);
    A << 1, dt, 0, 1;

    Eigen::MatrixXd B(n, 1);
    B << 0.5 * dt * dt, dt;

    Eigen::MatrixXd H(m, n);
    H << 1, 0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m) * 1.0;

    kf.setModel(A, B, H, Q, R);

    // 随机数生成器
    std::default_random_engine generator;
    std::normal_distribution<double> process_noise(0.0, 0.01);
    std::normal_distribution<double> measurement_noise(0.0, 1.0);

    // 初始真实状态
    Eigen::VectorXd true_x(n);
    true_x << 0.0, 0.0;

    // 控制输入，恒定加速度
    Eigen::VectorXd u(1);
    u << 1.0;

    // 画图所需的动态数组
    std::vector<double> time, true_vals, measured_vals, estimated_vals;

    std::cout << "Step\tTruePos\tMeasured\tEstimatedPos\tErrorMeasured\tErrorEstimated\n";
    for (int i = 0; i < 50; ++i) {
        Eigen::VectorXd process_noise_vec(n);
        for (int j = 0; j < n; ++j) {
            process_noise_vec(j) = process_noise(generator);
        }
        // 真实状态更新
        true_x = A * true_x + B * u + process_noise_vec;

        // 模拟观测
        Eigen::VectorXd z(m);
        z << true_x(0) + measurement_noise(generator);

        // 卡尔曼滤波器更新
        kf.predict(u);
        kf.update(z);
        Eigen::VectorXd est = kf.getState();
        double true_pos = true_x(0);
        double measured_pos = z(0);
        double estimated_pos = est(0);

        time.push_back(i);
        true_vals.push_back(true_pos);
        measured_vals.push_back(measured_pos);
        estimated_vals.push_back(estimated_pos);

        std::cout << std::setw(4) << i << std::setw(12) << true_pos << std::setw(12) << measured_pos
                  << std::setw(16) << estimated_pos << std::setw(16)
                  << (measured_pos - true_pos)                    // 测量误差
                  << std::setw(16) << (estimated_pos - true_pos)  // 估计误差
                  << "\n";
    }

    plt::figure_size(800, 600);
    plt::plot(time, true_vals, "g-");       // green line for true position
    plt::plot(time, measured_vals, "r.");   // red dots for measurements
    plt::plot(time, estimated_vals, "b-");  // blue line for estimates
    plt::xlabel("Step");
    plt::ylabel("Position");
    plt::title("Kalman Filter Position Tracking");
    plt::legend();
    plt::grid(true);
    plt::show();

    return 0;
}
