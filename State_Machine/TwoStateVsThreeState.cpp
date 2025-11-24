#include <cmath>
#include <iostream>
#include <string>
#include <vector>

enum class TwoState : size_t { Stable = 0, UnStable = 1 };
enum class ThreeState : size_t { Stable = 0, Candidate = 2, UnStable = 1 };

const char* toString(TwoState s) { return (s == TwoState::Stable ? "Stable" : "UnStable"); }

const char* toString(ThreeState s) {
    switch (s) {
        case ThreeState::Stable:
            return "Stable";
        case ThreeState::Candidate:
            return "Candidate";
        case ThreeState::UnStable:
            return "UnStable";
    }
    return "Unknown";
}

// ---------------- TwoState -------------------
struct TwoStateMachine {
    TwoState state = TwoState::Stable;
    int stable_count = 0, unstable_count = 0;

    int mk, nk;

    std::string selected_path;
    bool initialized = false;

    float thr = 0.5f;      // 小偏差认为稳定
    float big_thr = 1.0f;  // 大偏差认为不稳定

    TwoStateMachine(int m, int k, int n) : mk(m + k), nk(n + k) {}

    void update(const std::string& path, float deviation) {
        if (!initialized) {
            selected_path = path;
            initialized = true;
        }

        bool is_stable = deviation < thr;

        if (is_stable) {
            stable_count++;
            unstable_count = 0;
        } else {
            unstable_count++;
            stable_count = 0;
        }

        switch (state) {
            case TwoState::Stable:
                if (unstable_count >= mk) {
                    state = TwoState::UnStable;
                    selected_path = path;  // 切换路径
                }
                break;

            case TwoState::UnStable:
                if (stable_count >= nk) {
                    state = TwoState::Stable;
                    selected_path = path;  // 回来时也切换
                }
                break;
        }
    }
};

// ---------------- ThreeState -------------------
struct ThreeStateMachine {
    ThreeState state = ThreeState::Stable;
    int stable_count = 0, unstable_count = 0;

    int m, k, n;

    std::string selected_path;
    bool initialized = false;

    float thr = 0.5f;
    float big_thr = 1.0f;
    float small_thr = 0.2f;

    ThreeStateMachine(int m_, int k_, int n_) : m(m_), k(k_), n(n_) {}

    void update(const std::string& path, float deviation) {
        if (!initialized) {
            selected_path = path;
            initialized = true;
        }

        bool is_stable = deviation < thr;
        bool is_big = deviation > big_thr;      // 明显不稳定
        bool is_small = deviation < small_thr;  // 明显稳定

        if (is_stable) {
            stable_count++;
            unstable_count = 0;
        } else {
            unstable_count++;
            stable_count = 0;
        }

        switch (state) {
            case ThreeState::Stable:
                if (!is_stable && unstable_count >= m) {
                    state = ThreeState::Candidate;
                }
                break;

            case ThreeState::Candidate:
                if (is_big && unstable_count >= k) {
                    state = ThreeState::UnStable;
                    selected_path = path;  // 此时才切换！
                }
                if (is_small && stable_count >= k) {
                    state = ThreeState::Stable;
                }
                break;

            case ThreeState::UnStable:
                if (is_small && stable_count >= n) {
                    state = ThreeState::Candidate;
                }
                break;
        }
    }
};

// ---------------- main -------------------
int main() {
    std::vector<std::pair<std::string, float>> input = {{"path_1", 0.1},  {"path_2", 0.1},
                                                        {"path_3", 0.1},

                                                        {"path_4", 0.6},   // 不稳 1
                                                        {"path_5", 0.4},   // 稳 1
                                                        {"path_6", 0.7},   // 不稳 2
                                                        {"path_7", 0.3},   // 稳 2
                                                        {"path_8", 0.8},   // 不稳 3
                                                        {"path_9", 0.2},   // 稳 3
                                                        {"path_10", 0.9},  // 不稳 4

                                                        {"path_11", 0.1}, {"path_12", 0.2}};

    TwoStateMachine two(3, 2, 3);
    ThreeStateMachine three(3, 2, 3);

    std::cout << "Frame | path | dev | TwoState | Selected | ThreeState | Selected\n";
    std::cout << "--------------------------------------------------------------------------\n";

    for (size_t i = 0; i < input.size(); i++) {
        two.update(input[i].first, input[i].second);
        three.update(input[i].first, input[i].second);

        std::cout << i + 1 << " | " << input[i].first << " | " << input[i].second << " | "
                  << toString(two.state) << " | " << two.selected_path << " | "
                  << toString(three.state) << " | " << three.selected_path << "\n";
    }

    return 0;
}
