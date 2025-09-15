#include <memory>
#include <vector>
#include "DrivePoint.h"
#include "LineType.h"

struct DrivelineInstance {
   public:
    struct InstanceBranch {
        void Reset() {
            drivepoints.clear();
            freetouchdis = 0.f;
        }
        std::vector<DrivelineInstanceCell> drivepoints = {};
        float freetouchdis = 0.f;
    };

    struct InstanceBranchs {
        void Reset() {
            M.Reset();
            L.Reset();
            R.Reset();
            start_scanline_idx = SIZE_MAX;
        }
        InstanceBranch M;
        InstanceBranch L;
        InstanceBranch R;
        size_t start_scanline_idx;
    };

    /**
     * @brief 车道交通特征
     *
     */
    struct TrafficFeature {
        // 是否为辅路车道
        bool is_auxiliary = false;
        // 是否为逆向车道
        bool is_reverse = false;
        // 是否压过路沿
        bool cross_edge = false;
        // 行人占据率
        float pedestrian_occu_rate = 0.f;
        // 停止车辆占据率
        float parking_occu_rate = 0.f;
        // freespace的剩余距离
        float remain_distance = 0.f;
        // 是否可通行
        bool passable = true;
        // 是否设置为不可通行
        void SetUnpassable() { passable = false; }
        // 是否推荐通行
        bool Recommend() const {
            // 不是辅路/逆向/不压路沿/行人占据小于0.5/停车占据小于0.5/FS距离大于15m
            return !is_auxiliary && !is_reverse && !cross_edge && pedestrian_occu_rate < 0.5f &&
                   parking_occu_rate < 0.5f && remain_distance > 15.f;
        }
        // 转为字符串
        std::string ToString() {
            char desc[256];
            snprintf(desc, sizeof(desc),
                     "aux=%d reverse=%d cross_edge=%d peds=%.2f park=%.2f fs_dist=%.2f",
                     is_auxiliary, is_reverse, cross_edge, pedestrian_occu_rate, parking_occu_rate,
                     std::min(remain_distance, 1e3f));
            return std::string(desc);
        }
        void Reset() {
            is_auxiliary = false;
            // 是否为逆向车道
            is_reverse = false;
            // 是否压过路沿
            cross_edge = false;
            // 行人占据率
            pedestrian_occu_rate = 0.f;
            // 停止车辆占据率
            parking_occu_rate = 0.f;
            // freespace的剩余距离
            remain_distance = 0.f;
            // 是否可通行
            passable = true;
        }
    };

    void Reset() {
        isvalid = false;
        isdoublemode = false;
        left_similar = true;
        instance_branchs.Reset();
        vcs_pathpoints_L.clear();
        vcs_pathpoints_R.clear();
        x_splitdouble = -FLT_MAX;
        // 车道方向
        lane_direction = LaneDirection::LaneDirection_Unknown;
        // 车道交通特征
        traffic_feature.Reset();
        is_path_generated = false;
    }

    const Line<DrivePathPoint<float>> &GetSimilarBranch() const {
        if (left_similar) {
            return GetConstPathPointsL();
        } else {
            return GetConstPathPointsR();
        }
    }

    const Line<DrivePathPoint<float>> &GetConstPathPointsL() const { return vcs_pathpoints_L; }
    const Line<DrivePathPoint<float>> &GetConstPathPointsR() const { return vcs_pathpoints_R; }

    Line<DrivePathPoint<float>> &GetPathPointsL() { return vcs_pathpoints_L; }
    Line<DrivePathPoint<float>> &GetPathPointsR() { return vcs_pathpoints_R; }

    void SetFlag() {
        isdoublemode = instance_branchs.L.drivepoints.size() > 0UL &&
                       instance_branchs.R.drivepoints.size() > 0UL;

        isvalid = isdoublemode || instance_branchs.M.drivepoints.size() > 0UL;
        if (instance_branchs.M.drivepoints.size() > 0UL) {
            x_splitdouble = instance_branchs.M.drivepoints.back().drivepoint->scanline_ptr->x;
        } else {
            if (instance_branchs.L.drivepoints.size() > 0UL) {
                x_splitdouble = instance_branchs.L.drivepoints.front().drivepoint->scanline_ptr->x;
            }
        }
        // 车道方向
        bool has_backward = false;
        const float determine_x_range = 30.0f;
        for (auto &drive_pt : instance_branchs.M.drivepoints) {
            if (drive_pt.drivepoint == nullptr) {
                continue;
            }
            ScanLine *scanline_ptr = drive_pt.drivepoint->scanline_ptr;
            if (scanline_ptr == nullptr || std::fabs(scanline_ptr->x) > determine_x_range) {
                continue;
            }
            spScanPoint left_scan_pt = drive_pt.drivepoint->sub_left.lock();
            spScanPoint right_scan_pt = drive_pt.drivepoint->sub_right.lock();
            if ((left_scan_pt && left_scan_pt->direct_type == 3) ||
                (!left_scan_pt && right_scan_pt && right_scan_pt->direct_type == 3)) {
                has_backward = true;
                break;
            }
        }
        if (has_backward) {
            lane_direction = LaneDirection::LaneDirection_Backward;
        }
    }

    void GeneratePathPointsSelect() {
        SetFlag();
        GeneratePathPointsL();
        GeneratePathPointsR();
        // instance_branchs.Reset();
        is_path_generated = true;
    }

    bool PathGenerated() const { return is_path_generated; }

    DrivelineInstanceCell *GetDrivePointByScanlineidx(const size_t selectidx, const bool left);

    bool GetDrivePointByScanlineidx(const std::vector<spScanLine> &scanlines, size_t selectidx,
                                    bool left, spDrivePoint &drivept) const;

    bool GetScanlineidxByDrivePoint(const std::vector<spScanLine> &scanlines,
                                    const spDrivePoint &drive_pt, size_t &scanline_idx);
    /**
     * @brief Get the forward drive points by x value
     *
     * @param x
     * @param is_left left branch or right branch
     * @param forward_drive_pts
     * @return true forward drive points isn't empty
     * @return false forward drive points is empty
     */
    bool GetForwardDrivePointsByX(const float x, bool is_left,
                                  std::vector<DrivelineInstanceCell> &forward_drive_pts) const;

    bool TransformScanPoint2BoundryPoint(const spScanPoint scanpoint, BoundaryPoint &boundrypoint);

    static spScanLine GetScanLineByX(const std::vector<spScanLine> &scanlines, const float x) {
        if (scanlines.empty() || x < scanlines.front()->x - SCAN_GAP / 2.f ||
            x > scanlines.back()->x + SCAN_GAP / 2.f) {
            return nullptr;
        }

        int i = 0, j = static_cast<int>(scanlines.size()) - 1;
        // 循环，当搜索区间为空时跳出（当 i > j 时为空）
        while (i <= j) {
            int m = i + (j - i) / 2;  // 计算中点索引 m
            if (scanlines[static_cast<size_t>(m)]->x < x - SCAN_GAP / 2.f) {
                // 此情况说明 target 在区间 [m+1, j] 中
                i = m + 1;
            } else if (scanlines[static_cast<size_t>(m)]->x > x + SCAN_GAP / 2.f) {
                // 此情况说明 target 在区间 [i, m-1] 中
                j = m - 1;
            } else {
                // 找到目标元素，返回其索引
                return scanlines[static_cast<size_t>(m)];
            }
        }
        return nullptr;
    }

   public:
    void GeneratePathPointsByBranch(const std::vector<DrivelineInstanceCell> &driveline, float &s,
                                    Line<DrivePathPoint<float>> &vcs_pathpoints);

    void TransformTransition(const spDrivePoint &drive_point, DrivePathPoint<float> &drive_path_pt);

    void GenetateBoundryPointByDrivePoint(const spDrivePoint drivepoint, const bool is_left,
                                          BoundaryPoint &boundrypoint,
                                          BoundaryPoint &boundrypoint_real);

    void GeneratePathPointsL();

    void GeneratePathPointsR();

    /**
     * @brief 判断当前实例与邻居车道实例之间是否存在感知车道线/路沿
     * @param neighbor_instance 邻居车道实例
     * @return true 二者中间存在感知车道线/路沿
     * @return false 二者中间不存在感知车道线/路沿
     */
    bool HasUnVirtualLineBetweenInstances(DrivelineInstance &neighbor_instance) const;

    /**
     * @brief 判断指定的当前分支与邻居实例之间是否存在感知车道线/路沿
     * @param self_drivepoints 当前分支的驾驶点序列（M主干分支、左分支、右分支）
     * @param neighbor_instance 邻居车道实例
     * @return true 二者中间存在感知车道线/路沿
     * @return false 二者中间不存在感知车道线/路沿
     */
    bool HasUnVirtualLineBetweenBranchs(const std::vector<DrivelineInstanceCell> &self_drivepoints,
                                        DrivelineInstance &neighbor_instance) const;

   public:
    bool isvalid = false;
    bool isdoublemode = false;
    bool left_similar = true;
    bool is_path_generated = false;  // 表征是否生成过path

    float x_splitdouble = -FLT_MAX;
    SPLITREASON split_reason = SPLITREASON::NOSPLIT;
    // 车道方向
    LaneDirection lane_direction = LaneDirection::LaneDirection_Unknown;

    InstanceBranchs instance_branchs;
    // 车道交通特征
    TrafficFeature traffic_feature;

    // 判断和左邻居车道中间并不都是虚拟车道的点
    bool realline_exist_between_left_instance = false;
    // 判断和右邻居车道中间并不都是虚拟车道的点
    bool realline_exist_between_right_instance = false;

   private:
    Line<DrivePathPoint<float>> vcs_pathpoints_L;
    Line<DrivePathPoint<float>> vcs_pathpoints_R;
};
using spDrivelineInstance = std::shared_ptr<DrivelineInstance>;

inline bool LineTypeIsBoundry(LineType sidetype) {
    return sidetype == LineType::LINE_TYPE_UNKNOWN || sidetype == LineType::LINE_TYPE_CURB ||
           sidetype == LineType::LINE_TYPE_GUARDRAIL ||
           sidetype == LineType::LINE_TYPE_CONCRETEBARRIER ||
           sidetype == LineType::LINE_TYPE_FENCE || sidetype == LineType::LINE_TYPE_WALL ||
           sidetype == LineType::LINE_TYPE_CANOPY || sidetype == LineType::LINE_TYPE_CONE ||
           sidetype == LineType::LINE_TYPE_WATERHORSE ||
           sidetype == LineType::LINE_TYPE_GROUNDSIDE ||
           sidetype == LineType::LINE_TYPE_ROADEDGEUNKNOWN;
}

inline bool ScanpointBoundryTouch(const spScanPoint pt) {
    if (!pt) {
        return false;
    }
    return LineTypeIsBoundry(pt->line_type);
}

void GenerateLaterFakePoint(const spDrivePoint &this_thindrivept, bool searchnext,
                            spScanLine laterscanline, bool *continue_status,
                            std::vector<spScanPoint> &later_thin) {
    later_thin.resize(3UL);
    if (this_thindrivept == nullptr || laterscanline == nullptr) {
        return;
    }
    // create drive point
    later_thin[1] = std::make_shared<DrivePoint>();
    later_thin[1]->scanline_ptr = laterscanline.get();
    float this2later_gap = laterscanline->x - this_thindrivept->scanline_ptr->x;
    later_thin[1]->y = this2later_gap * tanf(this_thindrivept->theta) + this_thindrivept->y;
    later_thin[1]->theta = this_thindrivept->theta;
    spDrivePoint dript = std::dynamic_pointer_cast<DrivePoint>(later_thin[1]);
    dript->SetWidth(LANEWIDTH_VIRTUAL);

    continue_status[0] = false;
    continue_status[1] = false;
    later_thin[0] = this_thindrivept->sub_left.lock();
    later_thin[2] = this_thindrivept->sub_right.lock();
    // search left scan point
    if (later_thin[0]) {
        // forward search
        if (searchnext) {
            later_thin[0] = later_thin[0]->next.lock();
        } else {
            // backward search
            later_thin[0] = later_thin[0]->prev.lock();
        }
        if (!later_thin[0]) {
            // 如果在下一行找不到对应id的点，那么就找和他距离在1米之内的别的点
            spScanPoint this_left = this_thindrivept->sub_left.lock();
            float y = this2later_gap * tanf(this_left->theta) + this_left->y;
            for (size_t i = laterscanline->trafficlinepts.size() - 1UL;
                 i < laterscanline->trafficlinepts.size(); i--) {
                spScanPoint trafficpt = laterscanline->trafficlinepts[i];
                spScanPoint prev_trafficpt = trafficpt->prev.lock();
                if (!searchnext) {
                    prev_trafficpt = trafficpt->next.lock();
                }
                if (fabsf(y - trafficpt->y) < THRESHOLD_SAMEBOUNDRY) {
                    if (trafficpt->trafficline_ptr == this_left->trafficline_ptr ||
                        (prev_trafficpt && prev_trafficpt->y <= this_thindrivept->y)) {
                        continue;
                    }
                    if (ScanpointBoundryTouch(trafficpt)) {
                        if (trafficpt->y >= later_thin[1]->y) {
                            later_thin[0] = std::move(trafficpt);
                        }
                    } else {
                        later_thin[0] = trafficpt;
                    }
                    break;
                }
            }
        }
    }
    if (!later_thin[0]) {
        // not find left scan point, create virtual left scan point
        later_thin[0] = std::make_shared<ScanPoint>();
        later_thin[0]->scanline_ptr = laterscanline.get();
        float theta_compensate = fmaxf(cosf(later_thin[1]->theta), 0.707f);
        later_thin[0]->y = later_thin[1]->y + LANEWIDTH_VIRTUAL / 2.f / theta_compensate;
        if (std::isnan(later_thin[0]->y) || std::isinf(later_thin[0]->y)) {
            later_thin[0]->y = later_thin[1]->y + LANEWIDTH_VIRTUAL / 2.f;
        }
        later_thin[0]->theta = later_thin[1]->theta;
        // id can be different
        for (size_t i = laterscanline->trafficlinepts.size() - 1UL;
             i < laterscanline->trafficlinepts.size(); i--) {
            spScanPoint trafficpt = laterscanline->trafficlinepts[i];
            spScanPoint prev_trafficpt = trafficpt->prev.lock();
            if (!searchnext) {
                prev_trafficpt = trafficpt->next.lock();
            }
            float y = trafficpt->y;
            if (ScanpointBoundryTouch(trafficpt)) {
                if (prev_trafficpt && prev_trafficpt->y > this_thindrivept->y &&
                    trafficpt->y <= later_thin[1]->y) {
                    later_thin[0] = trafficpt;
                    continue_status[0] = true;
                    break;
                }
            } else if (y > later_thin[1]->y + THRESHOLD_SAMEBOUNDRY &&
                       y < (later_thin[0]->y + THRESHOLD_SAMEBOUNDRY)) {
                if (!prev_trafficpt || prev_trafficpt->y > this_thindrivept->y) {
                    later_thin[0] = trafficpt;
                    continue_status[0] = true;
                }
                break;
            }
        }
    } else {
        // find left scan point, left continue status is true
        continue_status[0] = true;
    }

    // search right scan point
    if (later_thin[2]) {
        // forward search
        if (searchnext) {
            later_thin[2] = later_thin[2]->next.lock();
        } else {
            // backward search
            later_thin[2] = later_thin[2]->prev.lock();
        }
        // id can be different
        if (!later_thin[2]) {
            spScanPoint this_right = this_thindrivept->sub_right.lock();
            float y = this2later_gap * tanf(this_right->theta) + this_right->y;
            for (size_t i = 0UL; i < laterscanline->trafficlinepts.size(); i++) {
                spScanPoint trafficpt = laterscanline->trafficlinepts[i];
                spScanPoint prev_trafficpt = trafficpt->prev.lock();
                if (!searchnext) {
                    prev_trafficpt = trafficpt->next.lock();
                }
                if (fabsf(trafficpt->y - y) < THRESHOLD_SAMEBOUNDRY) {
                    if (trafficpt->trafficline_ptr == this_right->trafficline_ptr ||
                        (prev_trafficpt && prev_trafficpt->y >= this_thindrivept->y)) {
                        continue;
                    }
                    if (ScanpointBoundryTouch(trafficpt)) {
                        if (trafficpt->y <= later_thin[1]->y) {
                            later_thin[2] = std::move(trafficpt);
                        }
                    } else {
                        later_thin[2] = trafficpt;
                    }
                    break;
                }
            }
        }
    }
    if (!later_thin[2]) {
        // not find right scan point, create virtual right scan point
        later_thin[2] = std::make_shared<ScanPoint>();
        later_thin[2]->scanline_ptr = laterscanline.get();
        float theta_compensate = fmaxf(cosf(later_thin[1]->theta), 0.707f);
        later_thin[2]->y = later_thin[1]->y - LANEWIDTH_VIRTUAL / 2.f / theta_compensate;
        if (std::isnan(later_thin[0]->y) || std::isinf(later_thin[0]->y)) {
            later_thin[2]->y = later_thin[1]->y - LANEWIDTH_VIRTUAL / 2.f;
        }
        later_thin[2]->theta = later_thin[1]->theta;
        // id can be different
        for (size_t i = 0UL; i < laterscanline->trafficlinepts.size(); i++) {
            spScanPoint trafficpt = laterscanline->trafficlinepts[i];
            spScanPoint prev_trafficpt = trafficpt->prev.lock();
            if (!searchnext) {
                prev_trafficpt = trafficpt->next.lock();
            }
            float y = trafficpt->y;
            if (ScanpointBoundryTouch(trafficpt)) {
                if (prev_trafficpt && prev_trafficpt->y < this_thindrivept->y &&
                    trafficpt->y >= later_thin[1]->y) {
                    later_thin[2] = trafficpt;
                    continue_status[1] = true;
                    break;
                }
            } else if (y < later_thin[1]->y - THRESHOLD_SAMEBOUNDRY &&
                       y > (later_thin[2]->y - THRESHOLD_SAMEBOUNDRY)) {
                if (!prev_trafficpt || prev_trafficpt->y < this_thindrivept->y) {
                    later_thin[2] = trafficpt;
                    continue_status[1] = true;
                }
                break;
            }
        }
    } else {
        // find right scan point, right continue status is true
        continue_status[1] = true;
    }
}

void FindLaterPointByOverlap(const spDrivePoint &this_thindrivept, const spScanLine laterscanline,
                             const std::vector<spScanPoint> &later_thin,
                             std::vector<std::pair<spDrivePoint, float>> &overlapdrivept) {
    for (spDrivePoint drivelinept : laterscanline->drivelinepts) {
        if (!drivelinept) {
            continue;
        }
        float left_boundry = fminf(drivelinept->sub_left.lock()->y, later_thin[0]->y);
        float right_boundry = fmaxf(drivelinept->sub_right.lock()->y, later_thin[2]->y);
        float overlap = left_boundry - right_boundry;
        if (overlap > canattach_lanewidth) {
            float dis = fabsf(this_thindrivept->y - right_boundry - overlap / 2.f);
            spDrivePoint tmppt = std::make_shared<DrivePoint>(*drivelinept);
            overlapdrivept.emplace_back(std::make_pair(tmppt, dis));
        }
    }
    std::sort(
        overlapdrivept.begin(), overlapdrivept.end(),
        [](const std::pair<spDrivePoint, float> &pair1,
           const std::pair<spDrivePoint, float> &pair2) { return pair1.second < pair2.second; });
}

bool UpdateSingleModeSearchPointByThinCenter(bool boundry_shift, const spDrivePoint hispoint,
                                             SELECTLIKE sl, const spDrivePoint this_thindrivept,
                                             bool searchnext, const spScanLine laterscanline,
                                             spDrivePoint &later_thindrivept) {
    later_thindrivept = nullptr;

    if (this_thindrivept == nullptr || laterscanline == nullptr) {
        return false;
    }
    // left line and right line continue status
    bool continue_status[2]{false, false};
    // size 3: left scan point, drive point, right scan point
    std::vector<spScanPoint> fake_later_thindrivepts(3UL, nullptr);
    GenerateLaterFakePoint(this_thindrivept, searchnext, laterscanline, continue_status,
                           fake_later_thindrivepts);

    std::vector<std::pair<spDrivePoint, float>> overlapdrivept;
    // 在laterscanline已有的drive point,
    // 查找与fake_later_thindrivepts有重叠的drive point
    FindLaterPointByOverlap(this_thindrivept, laterscanline, fake_later_thindrivepts,
                            overlapdrivept);

    bool meetclosed = this_thindrivept && this_thindrivept->is_closed;
    bool stopbyclosed =
        meetclosed && this_thindrivept->GetWidth() <= THRESHOLD_SAMEBOUNDRY && !searchnext;

    if (overlapdrivept.size() > 0UL && !stopbyclosed) {
        return CheckRoughDriveptBycontinuous(hispoint, sl, searchnext, overlapdrivept,
                                             this_thindrivept, fake_later_thindrivepts,
                                             later_thindrivept);
    }

    continue_status[0] = continue_status[0] && fake_later_thindrivepts[0];
    continue_status[1] = continue_status[1] && fake_later_thindrivepts[2];

    if (meetclosed && searchnext && this_thindrivept) {
        spScanPoint this_sub_left = this_thindrivept->sub_left.lock();
        spScanPoint this_sub_right = this_thindrivept->sub_right.lock();
        bool leftisboundry =
            ScanpointBoundryTouch(this_sub_left) || ScanpointMeetShadeArea(this_sub_left, 5.f);
        bool rightisboundry =
            ScanpointBoundryTouch(this_sub_right) || ScanpointMeetShadeArea(this_sub_right, 5.f);
        float this2later_gap = laterscanline->x - this_thindrivept->scanline_ptr->x;

        if (leftisboundry && !continue_status[0] && continue_status[1]) {
            float y = this2later_gap * tanf(this_sub_left->theta) + this_sub_left->y;
            float fakewidth = y - fake_later_thindrivepts[2]->y;
            if (this_thindrivept->GetWidth() <= LANEWIDTH_MIN / 2.f ||
                fakewidth <= LANEWIDTH_MIN / 2.f) {
                stopbyclosed = true;
            }
        } else if (rightisboundry && continue_status[0] && !continue_status[1]) {
            float y = this2later_gap * tanf(this_sub_right->theta) + this_sub_right->y;
            float fakewidth = fake_later_thindrivepts[0]->y - y;
            if (this_thindrivept->GetWidth() <= LANEWIDTH_MIN / 2.f ||
                fakewidth <= LANEWIDTH_MIN / 2.f) {
                stopbyclosed = true;
            }
        }
    }
    bool laterlanevalid = (laterscanline->x <= -SCAN_GAP * 0.5f && !searchnext) ||
                          (laterscanline->x >= SCAN_GAP * 0.5f && searchnext);
    if (continue_status[0] && continue_status[1]) {
        if (fake_later_thindrivepts[0]->y - fake_later_thindrivepts[2]->y <= 0.f ||
            fake_later_thindrivepts[0] == fake_later_thindrivepts[2]) {
            return false;
        }
        later_thindrivept = std::make_shared<DrivePoint>();
        later_thindrivept->sub_left = fake_later_thindrivepts[0];
        later_thindrivept->sub_right = fake_later_thindrivepts[2];
        later_thindrivept->SetWidth(fake_later_thindrivepts[0]->y - fake_later_thindrivepts[2]->y);
        later_thindrivept->theta =
            (fake_later_thindrivepts[0]->theta + fake_later_thindrivepts[2]->theta) / 2.f;
        later_thindrivept->y =
            (fake_later_thindrivepts[0]->y + fake_later_thindrivepts[2]->y) / 2.f;
        later_thindrivept->scanline_ptr = laterscanline.get();
        later_thindrivept->is_closed = true;
        // checkif meet the boundry
        if (boundry_shift) {
            ModeGenerateCommonInterface::CheckMeetBoundry(searchnext, this_thindrivept,
                                                          later_thindrivept);
        }
        return later_thindrivept != nullptr;
    } else if (continue_status[0]) {
        if (laterlanevalid && !stopbyclosed) {
            ModeGenerateCommonInterface::GenerateFakeDrivePtBySide(
                fake_later_thindrivepts[0], true, LANEWIDTH_VIRTUAL, later_thindrivept);
            // checkif meet the boundry
            if (boundry_shift) {
                ModeGenerateCommonInterface::CheckMeetBoundry(searchnext, this_thindrivept,
                                                              later_thindrivept);
            }
            return later_thindrivept != nullptr;
        }
        return false;
    } else if (continue_status[1]) {
        if (laterlanevalid && !stopbyclosed) {
            ModeGenerateCommonInterface::GenerateFakeDrivePtBySide(
                fake_later_thindrivepts[2], false, LANEWIDTH_VIRTUAL, later_thindrivept);
            // checkif meet the boundry
            if (boundry_shift) {
                ModeGenerateCommonInterface::CheckMeetBoundry(searchnext, this_thindrivept,
                                                              later_thindrivept);
            }
            return later_thindrivept != nullptr;
        }
        return false;
    }
    return false;
}

void CheckSingleModeForward(SELECTLIKE sl, size_t lastline_idx, spDrivePoint lastdrivept,
                            const std::vector<spScanLine> &scanlines,
                            std::vector<DrivelineInstanceCell> &drivepts) {
    drivepts.clear();
    spDrivePoint drivepoint_this;
    float freetouchgap = 0.f;
    for (size_t i = lastline_idx; i < scanlines.size(); i++) {
        if (i + 1UL >= scanlines.size() || lastdrivept == nullptr) {
            return;  // input error
        }
        float last2this = scanlines[i + 1UL]->x - scanlines[i]->x;
        if (fabsf(last2this) >= freetouch_threshold) {
            return;
        }
        spDrivePoint hispoint = nullptr;
        if (!UpdateSingleModeSearchPointByThinCenter(false, hispoint, sl, lastdrivept, true,
                                                     scanlines[i + 1UL], drivepoint_this) ||
            !drivepoint_this) {
            return;
        }

        if (drivepoint_this->sub_left.lock() == nullptr &&
            drivepoint_this->sub_right.lock() == nullptr) {
            if (freetouchgap + fabsf(last2this) > freetouch_threshold) {
                return;
            }
            freetouchgap += fabsf(last2this);
        } else {
            freetouchgap = 0.f;
        }
        DrivelineInstanceCell tmp;
        tmp.drivepoint = drivepoint_this;
        scanlines[i + 1UL]->GetBoundryScanPoint(tmp.drivepoint->y, tmp.fence_l, tmp.fence_r);
        scanlines[i + 1UL]->RefineBoundryScanPointByFs(tmp.drivepoint->y, tmp.fence_l, tmp.fence_r);
        tmp.scanlineidx = i + 1UL;
        drivepts.emplace_back(tmp);
        lastdrivept = drivepoint_this;
    }
}

spDrivelineInstance ConvertDrivePointToDriveLine(const spDrivePoint &drivepoint, SELECTLIKE mode,
                                                 const std::vector<spScanLine> &succ_scanlines) {
    // 生成 DrivelineInstance
    SimpleDriveInstanceGenerate mode_generator;
    spDrivelineInstance drive_inst = std::make_shared<DrivelineInstance>();
    std::vector<DrivelineInstanceCell> &drivepts = drive_inst->instance_branchs.M.drivepoints;
    mode_generator.CheckSingleModeForward(mode, 0, drivepoint, succ_scanlines, drivepts);
    DrivelineInstanceCell tmp;
    tmp.drivepoint = drivepoint;
    float y = drivepoint->y;
    succ_scanlines[0]->GetBoundryScanPoint(y, tmp.fence_l, tmp.fence_r);
    drivepts.insert(drivepts.begin(), tmp);
    drive_inst->isvalid = !drivepts.empty();
    drive_inst->isdoublemode = false;
    return drive_inst;
}