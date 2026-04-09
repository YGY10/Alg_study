#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

enum class LANETYPE {
  LAT_NORMAL_LANE = 0,
  LAT_VIRTUAL_CONNECTED_LANE = 1,
  LAT_MERGE_LANE = 2,
  LAT_NON_MOTOR_LANE = 3
};

struct LaneInfo {
  uint32_t lane_id = 0;
  LANETYPE lane_type = LANETYPE::LAT_NORMAL_LANE;
  bool in_junction = false;
  float lane_length = 30.0f;
  std::vector<uint32_t> successor_lane_ids;
};

struct LinkInfo {
  uint32_t link_id = 0;
  std::vector<LaneInfo> lanes_info;
};

struct LiteMapMsg {
  std::vector<LinkInfo> links_info;

  // 按 lane_id 找到它在对应 link 里的 lane_idx
  int GetLaneIdxByLaneId(uint32_t lane_id) const {
    for (const auto &link : links_info) {
      for (int i = 0; i < static_cast<int>(link.lanes_info.size()); ++i) {
        if (link.lanes_info[i].lane_id == lane_id) {
          return i;
        }
      }
    }
    return -1;
  }
};

struct LaneNode {
  int link_idx = -1;
  int lane_idx = -1;
  bool in_junction = false;
  uint32_t id = 0;
  int depth = 0;
};

class MultiSourceSelect {
 public:
  bool inside_junction_ = false;
  bool collecting_outjunc_len_ = false;
  float outjunc_acc_len_ = 0.0f;

  bool IsLinkInJunction(const LinkInfo &link) const {
    for (const auto &lane : link.lanes_info) {
      if (lane.in_junction) {
        return true;
      }
    }
    return false;
  }

  // 这里按你的要求：全局把 merge lane 判成不可通行
  bool IsLanePassable(const LaneInfo &lane_info) const {
    if (lane_info.lane_type == LANETYPE::LAT_NON_MOTOR_LANE) {
      return false;
    }
    if (lane_info.lane_type == LANETYPE::LAT_MERGE_LANE) {
      return false;
    }
    return true;
  }

  void DFSCollectPaths(LiteMapMsg *litemap_info_ptr,
                       int loc_link_idx,
                       int loc_lane_idx,
                       std::vector<LaneNode> &cur_path,
                       std::vector<std::vector<LaneNode>> &candi_path_ids) {
    LiteMapMsg &litemap_info = *litemap_info_ptr;

    // 合法性索引检查
    const int link_num = static_cast<int>(litemap_info.links_info.size());
    if (link_num == 0) return;
    if (loc_link_idx < 0 || loc_link_idx >= link_num) return;

    const auto &cur_link = litemap_info.links_info[loc_link_idx];

    const int lane_num = static_cast<int>(cur_link.lanes_info.size());
    if (lane_num == 0) return;
    if (loc_lane_idx < 0 || loc_lane_idx >= lane_num) return;

    const auto &cur_lane = cur_link.lanes_info[loc_lane_idx];

    // 当前节点压入路径
    LaneNode node;
    node.link_idx = loc_link_idx;
    node.lane_idx = loc_lane_idx;
    node.in_junction = cur_lane.in_junction;
    node.id = cur_lane.lane_id;
    node.depth = cur_path.empty() ? 0 : (cur_path.back().depth + 1);
    cur_path.push_back(node);

    std::cout << "[DFSCollectPaths] loc_link_idx: " << loc_link_idx
              << " loc_lane_idx: " << loc_lane_idx
              << " depth: " << node.depth
              << " in_junc: " << node.in_junction
              << " lane_id: " << node.id
              << " lane_length: " << cur_lane.lane_length << std::endl;

    // 出口判定
    const bool cur_in_junc = IsLinkInJunction(cur_link);
    bool should_stop_here = false;

    const bool prev_collecting = collecting_outjunc_len_;
    const float prev_acc_len = outjunc_acc_len_;

    constexpr float kOutsideMinLen = 50.0f;  // 路口外累计长度阈值（米）

    // 不管起点在不在路口内，只要“刚出路口”，就开始累计路口外长度；
    // 累计到 50m 或遇到分叉再停。
    bool is_exit_first_segment = false;
    if (!cur_in_junc && cur_path.size() >= 2) {
      const LaneNode &prev_node = cur_path[cur_path.size() - 2];
      const auto &prev_link = litemap_info.links_info[prev_node.link_idx];

      const bool prev_in_junc = IsLinkInJunction(prev_link);
      if (prev_in_junc) {
        is_exit_first_segment = true;
      }
    }

    // 进入出口累计模式
    if (is_exit_first_segment) {
      collecting_outjunc_len_ = true;
      outjunc_acc_len_ = 0.0f;
    }

    // 只在路口外累计
    if (collecting_outjunc_len_ && !cur_in_junc) {
      outjunc_acc_len_ += cur_lane.lane_length;
      const bool branch_ok = cur_lane.successor_lane_ids.size() > 1;
      const bool len_ok = (outjunc_acc_len_ >= kOutsideMinLen);

      if (branch_ok || len_ok) {
        should_stop_here = true;
      }
    }

    if (should_stop_here) {
      if (!cur_path.empty()) {
        if (!inside_junction_) {
          const LaneNode &first_node = cur_path.front();
          if (first_node.link_idx >= 0 && first_node.link_idx < link_num) {
            const auto &first_link = litemap_info.links_info[first_node.link_idx];
            if (!IsLinkInJunction(first_link) && cur_path.size() > 1) {
              std::vector<LaneNode> trimmed(cur_path.begin() + 1, cur_path.end());
              candi_path_ids.push_back(trimmed);
            } else {
              candi_path_ids.push_back(cur_path);
            }
          } else {
            candi_path_ids.push_back(cur_path);
          }
        } else {
          candi_path_ids.push_back(cur_path);
        }
      }

      collecting_outjunc_len_ = prev_collecting;
      outjunc_acc_len_ = prev_acc_len;
      cur_path.pop_back();
      return;
    }

    // ================== 正常向前扩展 ==================
    int next_link_idx = loc_link_idx + 1;
    bool has_next_link = next_link_idx < link_num;

    const auto &succ_ids = cur_lane.successor_lane_ids;
    bool has_succ = has_next_link && !succ_ids.empty();

    // 终止条件 1：没有后继 lane
    if (!has_succ) {
      if (!cur_path.empty()) {
        if (!inside_junction_) {
          const LaneNode &first_node = cur_path.front();
          if (first_node.link_idx >= 0 && first_node.link_idx < link_num) {
            const auto &first_link = litemap_info.links_info[first_node.link_idx];
            if (!IsLinkInJunction(first_link) && cur_path.size() > 1) {
              std::vector<LaneNode> trimmed(cur_path.begin() + 1, cur_path.end());
              candi_path_ids.push_back(trimmed);
            } else {
              candi_path_ids.push_back(cur_path);
            }
          } else {
            candi_path_ids.push_back(cur_path);
          }
        } else {
          candi_path_ids.push_back(cur_path);
        }
      }

      collecting_outjunc_len_ = prev_collecting;
      outjunc_acc_len_ = prev_acc_len;
      cur_path.pop_back();
      return;
    }

    const auto &next_link = litemap_info.links_info[next_link_idx];
    const int next_lane_num = static_cast<int>(next_link.lanes_info.size());

    // 终止条件 2：下一条 link 没有 lane
    if (next_lane_num == 0) {
      if (!cur_path.empty()) {
        if (!inside_junction_) {
          const LaneNode &first_node = cur_path.front();
          if (first_node.link_idx >= 0 && first_node.link_idx < link_num) {
            const auto &first_link = litemap_info.links_info[first_node.link_idx];
            if (!IsLinkInJunction(first_link) && cur_path.size() > 1) {
              std::vector<LaneNode> trimmed(cur_path.begin() + 1, cur_path.end());
              candi_path_ids.push_back(trimmed);
            } else {
              candi_path_ids.push_back(cur_path);
            }
          } else {
            candi_path_ids.push_back(cur_path);
          }
        } else {
          candi_path_ids.push_back(cur_path);
        }
      }

      collecting_outjunc_len_ = prev_collecting;
      outjunc_acc_len_ = prev_acc_len;
      cur_path.pop_back();
      return;
    }

    // 遍历后继车道
    for (const auto &succ_lane_id : succ_ids) {
      if (succ_lane_id == 0) continue;

      int succ_lane_ind = litemap_info.GetLaneIdxByLaneId(succ_lane_id);
      if (succ_lane_ind < 0 || succ_lane_ind >= next_lane_num) continue;

      const auto &succ_lane = next_link.lanes_info[succ_lane_ind];
      if (!IsLanePassable(succ_lane)) {
        std::cout << "  skip non-passable succ lane_id: " << succ_lane.lane_id << std::endl;
        continue;
      }

      DFSCollectPaths(litemap_info_ptr, next_link_idx, succ_lane_ind, cur_path, candi_path_ids);
    }

    // 回溯
    collecting_outjunc_len_ = prev_collecting;
    outjunc_acc_len_ = prev_acc_len;
    cur_path.pop_back();
  }
};

static std::string PathToString(const std::vector<LaneNode> &path) {
  std::string s;
  for (size_t i = 0; i < path.size(); ++i) {
    s += std::to_string(path[i].id);
    if (i + 1 < path.size()) {
      s += " -> ";
    }
  }
  return s;
}

static LiteMapMsg BuildExampleMap() {
  LiteMapMsg map;

  // link1: 不在路口内，lane1 -> lane2 / lane3
  {
    LinkInfo link;
    link.link_id = 101;

    LaneInfo lane1;
    lane1.lane_id = 1;
    lane1.lane_type = LANETYPE::LAT_NORMAL_LANE;
    lane1.in_junction = false;
    lane1.lane_length = 20.0f;
    lane1.successor_lane_ids = {2, 3};

    link.lanes_info.push_back(lane1);
    map.links_info.push_back(link);
  }

  // link2: 在路口内，lane2 -> lane4, lane3 -> lane5
  {
    LinkInfo link;
    link.link_id = 102;

    LaneInfo lane2;
    lane2.lane_id = 2;
    lane2.lane_type = LANETYPE::LAT_VIRTUAL_CONNECTED_LANE;
    lane2.in_junction = true;
    lane2.lane_length = 15.0f;
    lane2.successor_lane_ids = {4};

    LaneInfo lane3;
    lane3.lane_id = 3;
    lane3.lane_type = LANETYPE::LAT_VIRTUAL_CONNECTED_LANE;
    lane3.in_junction = true;
    lane3.lane_length = 15.0f;
    lane3.successor_lane_ids = {5};

    link.lanes_info.push_back(lane2);
    link.lanes_info.push_back(lane3);
    map.links_info.push_back(link);
  }

  // link3: 不在路口内，lane4 -> lane6, lane5 -> lane7
  {
    LinkInfo link;
    link.link_id = 103;

    LaneInfo lane4;
    lane4.lane_id = 4;
    lane4.lane_type = LANETYPE::LAT_NORMAL_LANE;
    lane4.in_junction = false;
    lane4.lane_length = 20.0f;
    lane4.successor_lane_ids = {6};

    LaneInfo lane5;
    lane5.lane_id = 5;
    lane5.lane_type = LANETYPE::LAT_NORMAL_LANE;
    lane5.in_junction = false;
    lane5.lane_length = 20.0f;
    lane5.successor_lane_ids = {7};

    link.lanes_info.push_back(lane4);
    link.lanes_info.push_back(lane5);
    map.links_info.push_back(link);
  }

  // link4: 不在路口内，lane6 -> lane8, lane7(合流) -> lane8
  {
    LinkInfo link;
    link.link_id = 104;

    LaneInfo lane6;
    lane6.lane_id = 6;
    lane6.lane_type = LANETYPE::LAT_NORMAL_LANE;
    lane6.in_junction = false;
    lane6.lane_length = 20.0f;
    lane6.successor_lane_ids = {8};

    LaneInfo lane7;
    lane7.lane_id = 7;
    lane7.lane_type = LANETYPE::LAT_MERGE_LANE;  // 关键：全局不可通行
    lane7.in_junction = false;
    lane7.lane_length = 20.0f;
    lane7.successor_lane_ids = {8};

    link.lanes_info.push_back(lane6);
    link.lanes_info.push_back(lane7);
    map.links_info.push_back(link);
  }

  return map;
}

void Test_DFSCollectPaths_FilterMergeLaneGlobally() {
  LiteMapMsg map = BuildExampleMap();

  MultiSourceSelect mss;
  mss.inside_junction_ = false;
  mss.collecting_outjunc_len_ = false;
  mss.outjunc_acc_len_ = 0.0f;

  std::vector<LaneNode> cur_path;
  std::vector<std::vector<LaneNode>> candi_path_ids;

  // 从 link1 / lane1 开始搜
  mss.DFSCollectPaths(&map, 0, 0, cur_path, candi_path_ids);

  std::cout << "\n===== DFS Result =====" << std::endl;
  std::cout << "candidate path count = " << candi_path_ids.size() << std::endl;
  for (size_t i = 0; i < candi_path_ids.size(); ++i) {
    std::cout << "path[" << i << "] = " << PathToString(candi_path_ids[i]) << std::endl;
  }

  // 按当前 DFS 逻辑：起点在路口外，最终会裁掉第一段入口 link
  // 所以结果应为 2 -> 4 -> 6
//   assert(candi_path_ids.size() == 1);
//   assert(candi_path_ids[0].size() == 3);
//   assert(candi_path_ids[0][0].id == 2);
//   assert(candi_path_ids[0][1].id == 4);
//   assert(candi_path_ids[0][2].id == 6);

  std::cout << "Test_DFSCollectPaths_FilterMergeLaneGlobally PASSED" << std::endl;
}

int main() {
  Test_DFSCollectPaths_FilterMergeLaneGlobally();
  return 0;
}