#include <array>
#include <iostream>

enum State : int { NORMAL = 0, DANGER = 1, LANECHANGE = 2, COOL = 3 };
enum TargetLaneIndex : int { LANE0 = 0, LANE1 = 1, LANE2 = 2 };

struct LaneChangeInfo {
    State state;
    TargetLaneIndex target_lane_index;
    float target_speed;
    float block_time;
};

LaneChangeInfo LaneChangeStateMachine(LaneChangeInfo last_lane_change_info,
                                      std::array<bool, 3> is_danger, bool can_brake, float time,
                                      float recommond_v, std::array<float, 2> ava_max_acc = {2, -3},
                                      float dt = 0.05) {
    LaneChangeInfo output = last_lane_change_info;
    switch (last_lane_change_info.target_lane_index) {
        case LANE0:
            switch (last_lane_change_info.state) {
                case NORMAL:
                    output.block_time = 0.f;
                    if (is_danger[0]) {  // 跳转到危险状态处理
                        output.state = DANGER;
                    } else {  // 不危险，正常加速, 其他不变
                        output.target_speed =
                            std::min(output.target_speed + ava_max_acc[0] * dt, recommond_v);
                    }
                    break;

                case DANGER:
                    if (is_danger[0]) {
                        if (can_brake) {  // 能通过刹车避障, 开始减速
                            output.target_speed = std::max(
                                last_lane_change_info.target_speed + ava_max_acc[1] * dt, 1.f);
                            output.block_time = time + 2 * dt;
                        } else {
                            if (!is_danger[1]) {  // 1车道不危险，向1车道变道, 车速保持不变
                                output.state = LANECHANGE;
                                output.target_lane_index = LANE1;
                            }
                        }
                    } else {
                        // 危险结束，进入保持阶段
                        output.state = COOL;
                    }
                    break;

                case LANECHANGE:
                    output.block_time = time + 1 * dt;
                    output.state = COOL;
                    // 速度不变
                    break;

                case COOL:
                    if (time > last_lane_change_info.block_time) {
                        output.state = NORMAL;
                    }
                    break;
                default:
                    break;
            }

            break;
        case LANE1:
            switch (last_lane_change_info.state) {
                case NORMAL:
                    output.block_time = 0.f;
                    if (is_danger[1]) {
                        output.state = DANGER;
                    } else {
                        // 不危险，正常加速, 其他不变
                        output.target_speed =
                            std::min(output.target_speed + ava_max_acc[0] * dt, recommond_v);
                    }
                    break;
                case DANGER:
                    if (is_danger[1]) {
                        if (can_brake) {  // 能通过刹车避障, 开始减速
                            output.target_speed = std::max(
                                last_lane_change_info.target_speed + ava_max_acc[1] * dt, 1.f);
                            output.block_time = time + 2 * dt;
                        } else {
                            if (!is_danger[0]) {  // 0车道不危险，向1车道变道, 车速保持不变
                                output.state = LANECHANGE;
                                output.target_lane_index = LANE0;
                            } else if (!is_danger[2]) {  // 2车道不危险， 向2车道变道， 车速保持不变
                                output.state = LANECHANGE;
                                output.target_lane_index = LANE2;
                            }
                        }
                    } else {
                        // 危险结束，进入保持阶段
                        output.state = COOL;
                    }
                    break;

                case LANECHANGE:
                    output.block_time = time + 1 * dt;
                    output.state = COOL;
                    // 速度不变
                    break;

                case COOL:
                    if (time >= last_lane_change_info.block_time) {
                        output.state = NORMAL;
                    }
                    break;
                default:
                    break;
            }
            break;

        case LANE2:
            switch (last_lane_change_info.state) {
                case NORMAL:
                    output.block_time = 0.f;
                    if (is_danger[2]) {  // 跳转到危险状态处理
                        output.state = DANGER;
                    } else {  // 不危险，正常加速, 其他不变
                        output.target_speed =
                            std::min(output.target_speed + ava_max_acc[0] * dt, recommond_v);
                    }
                    break;

                case DANGER:
                    if (is_danger[2]) {
                        if (can_brake) {  // 能通过刹车避障, 开始减速
                            output.target_speed = std::max(
                                last_lane_change_info.target_speed + ava_max_acc[1] * dt, 1.f);
                            output.block_time = time + 2 * dt;
                        } else {
                            if (!is_danger[1]) {  // 1车道不危险，向1车道变道, 车速保持不变
                                output.state = LANECHANGE;
                                output.target_lane_index = LANE1;
                            }
                        }
                    } else {
                        // 危险结束，进入保持阶段
                        output.state = COOL;
                    }
                    break;

                case LANECHANGE:
                    output.block_time = time + 1 * dt;
                    output.state = COOL;
                    // 速度不变
                    break;

                case COOL:
                    if (time > last_lane_change_info.block_time) {
                        output.state = NORMAL;
                    }
                    break;
                default:
                    break;
            }
            break;

        default:
            break;
    }
    return output;
};

int main() {
    LaneChangeInfo lc{NORMAL, LANE1, 10.0f, 0.0f};

    float time = 0.0;

    for (int i = 0; i < 8; i++) {
        // 输入构造
        std::array<bool, 3> danger = {false, false, false};
        if (i <= 3) danger[1] = true;

        bool can_brake = false;
        float recommond_v = 15;

        // 记录输入（状态机调用前的 old 值）
        State in_state = lc.state;
        TargetLaneIndex in_lane = lc.target_lane_index;
        float in_speed = lc.target_speed;
        float in_block = lc.block_time;

        // 状态机输出
        lc = LaneChangeStateMachine(lc, danger, can_brake, time, recommond_v);

        // 一行打印 输入 + 输出
        printf(
            "t=%.2f  in_state=%d  in_lane=%d  in_speed=%.2f  in_block=%.2f  "
            "danger=[%d %d %d]  brake=%d  "
            "out_state=%d  out_lane=%d  out_speed=%.2f  out_block=%.2f\n",
            time, in_state, in_lane, in_speed, in_block, danger[0], danger[1], danger[2], can_brake,
            lc.state, lc.target_lane_index, lc.target_speed, lc.block_time);

        time += 0.05;
    }
}
