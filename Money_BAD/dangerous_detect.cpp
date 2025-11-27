#include <Python.h>
#include <iostream>
#include <vector>
#include "include/matplotlibcpp.h"
#include <cmath>
#include <limits>

namespace plt = matplotlibcpp;

class Actor {
public:
    float x = 0.f;
    float y = 0.f;
    float v = 0.f;
    float heading = 0.f; // rad
};

struct Vec2 {
    float x, y;
};

static inline float cross(const Vec2& a, const Vec2& b) {
    return a.x * b.y - a.y * b.x;
}

static inline float dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}



bool dangerous_detect(const Actor& a1, const Actor& a2, Vec2 &collision_pose) {
    Vec2 P1{a1.x, a1.y};
    Vec2 P2{a2.x, a2.y};

    Vec2 D1{a1.v * std::cos(a1.heading), a1.v * std::sin(a1.heading)};
    Vec2 D2{a2.v * std::cos(a2.heading), a2.v * std::sin(a2.heading)};

    Vec2 r{P2.x - P1.x, P2.y - P1.y};

    float denom = cross(D1, D2);

    //---------- 情况1：速度方向平行/反平行（D1 || D2）----------//
    if (std::fabs(denom) < 1e-6) {
        // 不共线 => 不可能碰撞
        if (std::fabs(cross(r, D1)) >= 1e-6)
            return false;

        // 共线情况下：需要判断双方速度是否接近
        Vec2 v_rel{D1.x - D2.x, D1.y - D2.y};  // 相对速度

        float dist = std::sqrt(r.x*r.x + r.y*r.y);
        float v_rel_mag = std::sqrt(v_rel.x*v_rel.x + v_rel.y*v_rel.y);

        if (v_rel_mag < 1e-6)
            return false; // 相对静止，不会靠近

        // 相对距离缩短 => 有可能碰撞
        float t = - (r.x*v_rel.x + r.y*v_rel.y) / (v_rel_mag*v_rel_mag);

        if (t >= 0) {
            collision_pose.x = P1.x + D1.x * t;
            collision_pose.y = P1.y + D1.y * t;
            return true;
        }
        return false;
    }

    //---------- 情况2：速度方向不平行（正常 case）----------//
    float t1 = cross(r, D2) / denom;
    float t2 = cross(r, D1) / denom;

    if (t1 >= 0 && t2 >= 0) {
        collision_pose.x = P1.x + t1 * D1.x;
        collision_pose.y = P1.y + t1 * D1.y;
        return true;
    }

    return false;
}


int main() {
    std::cout << "Matplotlib-cpp test start\n";

    // 告诉 Python 去哪里找系统包（和 python3 -c 的 sys.path 完全一致）
    setenv("PYTHONPATH",
        "/usr/lib/python3.8:"
        "/usr/lib/python3.8/lib-dynload:"
        "/usr/lib/python3/dist-packages",
        1);

    // 初始化 Python
    auto &interp = plt::detail::_interpreter::get();

    // 打印 sys.path
    PyRun_SimpleString(R"(
import sys
print("C++ embedded sys.path =", sys.path)
)");
    std::array<float, 3> lane_y = {7.f, 3.5f, 0.f};
    // 测试 Actor
    Actor A{0.f, 5.f, 1.f, 0.f};           // 从 (0,0) 朝 +x 方向走
    Actor B{20.f, 5.f, 2.f, 3.14};       // 从 (5,5) 朝 -y 方向走

    Vec2 collision_pose;
    bool collide = dangerous_detect(A, B, collision_pose);

    std::cout << "Collision? " << collide 
              << " collision pose = ( " << collision_pose.x << " , " << collision_pose.y << " )\n";

    // ---- 可视化 ----
    std::vector<double> ax, ay, bx, by;

    for (int i = 0; i <= 50; i++) {
        float t = i * 0.1f;

        ax.push_back(A.x + A.v * std::cos(A.heading) * t);
        ay.push_back(A.y + A.v * std::sin(A.heading) * t);

        bx.push_back(B.x + B.v * std::cos(B.heading) * t);
        by.push_back(B.y + B.v * std::sin(B.heading) * t);
    }

    // 1) A、B 的轨迹
    plt::plot(ax, ay, "r-"); // A 轨迹（红）
    plt::plot(bx, by, "b-"); // B 轨迹（蓝）

    // 2) 初始点
    std::vector<double> A0x{A.x}, A0y{A.y};
    std::vector<double> B0x{B.x}, B0y{B.y};

    plt::scatter(A0x, A0y, 80.0, {{"color","black"}});
    plt::scatter(B0x, B0y, 80.0, {{"color","black"}});

    // 3) 碰撞点（如果会碰撞）
    if (collide) {
        double cx = collision_pose.x;
        double cy = collision_pose.y;

        std::vector<double> cxv{cx};
        std::vector<double> cyv{cy};

        plt::scatter(cxv, cyv, 120.0, {{"color","red"}});
    }

    // 4) 设置坐标等比
    plt::axis("equal");
    plt::title("Dangerous Detect Ray Intersection with Initial Points");

    plt::show();

}
