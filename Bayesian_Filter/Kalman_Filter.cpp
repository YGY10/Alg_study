#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// 卡尔曼滤波器类（1D）
class KalmanFilter {
   public:
    KalmanFilter(double init_state, double init_uncertainty, double process_noise,
                 double measurement_noise, double observation_matrix)
        : x_est(init_state),
          P(init_uncertainty),
          Q(process_noise),
          R(measurement_noise),
          H(observation_matrix) {}

    // 更新预测
    void update(double measurement) {
        // 预测
        double x_pred = x_est;  // 状态预测（无控制输入，匀速）
        double P_pred = P + Q;  // 预测协方差

        // 更新（卡尔曼增益）
        double K = P_pred * H / (H * P_pred * H + R);

        // 融合观测
        x_est = x_pred + K * (measurement - H * x_pred);
        P = (1 - K * H) * P_pred;
    }

    double get_estimate() const { return x_est; }

   private:
    double x_est;  // 当前状态估计
    double P;      // 当前估计不确定性
    double Q;      // 过程噪声
    double R;      // 观测噪声
    double H;      // 观测矩阵
};

int main() {
    const int steps = 50;
    const double true_velocity = 1.0;  // 匀速运动
    const double process_noise_std = 0.1;
    const double measurement_noise_std = 1.0;

    // 随机数生成器
    std::default_random_engine generator;
    std::normal_distribution<double> process_noise(0.0, process_noise_std);
    std::normal_distribution<double> measurement_noise(0.0, measurement_noise_std);

    // 初始化
    double true_position = 0.0;
    KalmanFilter kf(0.0, 1.0, process_noise_std * process_noise_std,
                    measurement_noise_std * measurement_noise_std, 1.0);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Step\tTrue\tMeasure\tEstimate\n";

    for (int i = 0; i < steps; ++i) {
        // 模拟真实位置（加一点过程噪声）
        true_position += true_velocity + process_noise(generator);

        // 模拟观测（加入观测噪声）
        double measurement = true_position + measurement_noise(generator);

        // 卡尔曼滤波更新
        kf.update(measurement);

        // 输出
        std::cout << i << "\t" << true_position << "\t" << measurement << "\t" << kf.get_estimate()
                  << "\n";
    }

    return 0;
}
