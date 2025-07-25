// 粒子滤波
// 粒子滤波器是一种使用蒙特卡罗方法的递归滤波器，透过一组具有权重的随机样本（称为粒子）来表示随机事件的后验概率，从含有噪声或不完整的观测序列，估计出动态系统的状态
// 粒子滤波器可以用在任何状态空间模型上
// 粒子滤波器是卡尔曼滤波器的一般化方法，卡尔曼滤波器建立在线性的状态空间和高斯分布的噪声上
// 而粒子滤波器的状态空间模型可以是非线性，且噪声分布可以是任何型式
// 为什么要用粒子滤波？
// KF: 要求系统是线性，高斯噪声
// EKF: 系统是弱非线性，近似化线性可行
// UKF: 系统中等线性，但仍假设噪声是高斯分布
// 显示系统常常是： 非线性 非高斯
#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include "../eigen-3.4.0/Eigen/Dense"

class ParticleFilter {
   public:
    ParticleFilter(int num_particles, double meas_std)
        : N(num_particles),
          meas_std(meas_std),
          particles(num_particles, Eigen::Vector2d::Zero()),
          weights(num_particles, 1.0 / num_particles) {}

    void init(const Eigen::Vector2d& init_state, std::default_random_engine& gen,
              std::normal_distribution<double>& noise) {
        for (int i = 0; i < N; ++i) {
            particles[i] = init_state;
            for (int j = 0; j < 2; ++j) {
                particles[i](j) += noise(gen);
            }
        }
    }

    void predict(
        double control_input, std::default_random_engine& gen,
        std::normal_distribution<double>& noise,
        const std::function<Eigen::Vector2d(const Eigen::Vector2d&, double)>& transition_model) {
        for (int i = 0; i < N; ++i) {
            particles[i] = transition_model(particles[i], control_input);
            for (int j = 0; j < 2; ++j) {
                particles[i](j) += noise(gen);
            }
        }
    }

    void update(double z) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            double error = z - particles[i](0);
            weights[i] = std::exp(-0.5 * (error * error) / (meas_std * meas_std + 1e-6));
            sum += weights[i];
        }
        for (int i = 0; i < N; ++i) {
            weights[i] /= (sum + 1e-6);
        }
    }

    void resample(std::default_random_engine& gen) {
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        std::vector<Eigen::Vector2d> new_particles;
        for (int i = 0; i < N; ++i) {
            new_particles.push_back(particles[dist(gen)]);
        }
        particles = std::move(new_particles);
        std::fill(weights.begin(), weights.end(), 1.0 / N);
    }

    Eigen::Vector2d estimate() const {
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        for (int i = 0; i < N; ++i) {
            mean += weights[i] * particles[i];
        }
        return mean;
    }

   private:
    int N;
    double meas_std;
    std::vector<Eigen::Vector2d> particles;
    std::vector<double> weights;
};

#endif  // PARTICLE_FILTER_H
