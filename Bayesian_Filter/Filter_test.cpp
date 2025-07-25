#include <iomanip>  // 添加 setw 所需的头文件
#include "../plot/matplotlibcpp.h"
#include "Bayesian_Filter/Particle_Filter.cpp"

namespace plt = matplotlibcpp;
int main() {
    int n = 2;  // 状态维度：位置 + 速度
    int m = 1;  // 观测维度：位置
    int l = 1;  // 控制输入维度：加速度
    double dt = 1.0;

    Eigen::Matrix2d A;
    A << 1, dt, 0, 1;
    Eigen::Vector2d B;
    B << 0.5 * dt * dt, dt;

    // 初始状态
    Eigen::VectorXd x0(n);
    x0 << 0.0, 0.0;

    ParticleFilter pf(1000, 1.0);
    std::default_random_engine generator;
    std::normal_distribution<double> process_noise(0.0, 0.01);
    std::normal_distribution<double> measurement_noise(0.0, 1.0);
    std::normal_distribution<double> init_noise(0.0, 1.0);
    pf.init(x0, generator, init_noise);

    // 状态转移模型
    auto transition = [dt](const Eigen::Vector2d& x, double u) {
        Eigen::Matrix2d A;
        A << 1, dt, 0, 1;
        Eigen::Vector2d B;
        B << 0.5 * dt * dt, dt;
        return A * x + B * u;
    };

    Eigen::VectorXd true_x(n);
    true_x << 0.0, 0.0;
    Eigen::VectorXd u(l);
    u << 1.0;

    std::vector<double> time, true_vals, measured_vals, pf_vals;

    std::cout << "Step\tTrue\tMeasured\tPF_Est\tError_PF\n";
    for (int i = 0; i < 50; ++i) {
        // 模拟真实状态
        Eigen::VectorXd process_noise_vec(n);
        for (int j = 0; j < n; ++j) {
            process_noise_vec(j) = process_noise(generator);
        }
        true_x = A * true_x + B * u + process_noise_vec;

        // 观测
        Eigen::VectorXd z(m);
        z << true_x(0) + measurement_noise(generator);

        // 粒子滤波器
        pf.predict(u(0), generator, process_noise, transition);
        pf.update(z(0));
        Eigen::Vector2d pf_est = pf.estimate();
        pf.resample(generator);

        // 存储并输出
        double t = i;
        double gt = true_x(0);
        double meas = z(0);
        // double kf_pos = kf_est(0);
        double pf_pos = pf_est(0);

        time.push_back(t);
        true_vals.push_back(gt);
        measured_vals.push_back(meas);
        // kf_vals.push_back(kf_pos);
        pf_vals.push_back(pf_pos);

        std::cout << std::setw(4) << i << std::setw(10) << gt << std::setw(12) << meas
                  << std::setw(12) << pf_pos << std::setw(12) << (pf_pos - gt) << "\n";
    }

    // 画图
    // plt::figure_size(900, 600);
    // plt::plot(time, true_vals, "g-");      // green line for true position
    // plt::plot(time, measured_vals, "r.");  // red dots for measurements
    // plt::plot(time, pf_vals, "b-");        // blue line for estimates
    // plt::xlabel("Step");
    // plt::ylabel("Position");
    // plt::title("Particle Filter Position Tracking");
    // plt::legend();
    // plt::grid(true);
    // plt::show();

    return 0;
}