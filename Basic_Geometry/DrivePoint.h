struct ScanPoint {
    virtual ~ScanPoint() {}

    static void CalculateTheta(const std::shared_ptr<ScanPoint> ptnext,
                               const std::shared_ptr<ScanPoint> ptprev,
                               std::shared_ptr<ScanPoint> ptmid);

    void SetInfo(const HOBOT_NOA::TrafficLine *trafficline, float x) {
        if (!trafficline) {
            return;
        }
        line_type = static_cast<HOBOT_SSM::LineType>(trafficline->ori_line_type);
        marking_type = static_cast<LineMarking>(
            trafficline->get_value_from_pairs(trafficline->ori_markings, x));
        line_source = trafficline->ori_line_source;
        color_type =
            static_cast<LineColor>(trafficline->get_value_from_pairs(trafficline->ori_colors, x));
        direct_type = trafficline->direct_type;
        if (direct_type == 0) {
            direct_type = 1;
        }
        // 如果是黄色车道线，设置direct type为双向
        if (color_type == LineColor::LINE_COLOR_YELLOW) {
            direct_type = 2;
        }
        // 只对车道线进行逆向判断
        const Line<LinePoint<float>> &pts = trafficline->line;
        bool in_order = pts.back().x >= pts.front().x;
        if (line_type == LINE_TYPE_LANELINE && !in_order && direct_type != 2) {
            direct_type = 3;  // 逆向
        }
        edge_side = trafficline->edge_side;
        trafficline_ptr = const_cast<HOBOT_NOA::TrafficLine *>(trafficline);
        int type = 0;
        float threshold = SCAN_GAP / 2.f + 0.1f;
        for (const auto &trextra_point : trafficline_ptr->extra_points) {
            bool division = trextra_point.point_type &
                            TO_INT(HOBOT_NOA::ExtraPointType::EXTRA_POINT_TYPE_DIVISION);
            bool merge = trextra_point.point_type &
                         TO_INT(HOBOT_NOA::ExtraPointType::EXTRA_POINT_TYPE_MERGE);
            if (!division && !merge) {
                continue;
            }
            if (fabsf(trextra_point.point.x - x) < threshold) {
                type += trextra_point.point_type;
            }
        }
        extra_point_type = type;
    }

    template <typename T1, typename T2>
    void GenerateByPoints(const T1 &pt0, const T1 &pt1, const std::shared_ptr<T2> scanline) {
        if (!scanline) {
            return;
        }
        y = (pt1.y - pt0.y) / (pt1.x - pt0.x) * (scanline->x - pt0.x) + pt0.y;
        if (std::isnan(y) || std::isinf(y)) {
            y = pt1.y;
        }
        theta = atan2(pt1.y - pt0.y, pt1.x - pt0.x);
        scanline_ptr = scanline.get();
    }

    float y = 0.f;
    float theta = 0.f;
    ScanLine *scanline_ptr = nullptr;

    LineType line_type = LineType::LINE_TYPE_UNKNOWN;
    LineMarking marking_type = LineMarking::LINE_MARKING_UNKNOWN;
    uint64_t line_source = 0UL;
    LineColor color_type = LineColor::LINE_COLOR_UNKNOWN;
    int direct_type = 1;  // 1: 单向, 2: 双向, 3: 逆向
    EdgeSide edge_side = EdgeSide::EDGE_SIDE_UNKNOWN;
    HOBOT_NOA::TrafficLine *trafficline_ptr = nullptr;
    int extra_point_type = static_cast<int>(HOBOT_NOA::ExtraPointType::EXTRA_POINT_TYPE_UNSET);

    std::weak_ptr<ScanPoint> prev;
    std::weak_ptr<ScanPoint> next;
    bool rendered = false;
};
using spScanPoint = std::shared_ptr<ScanPoint>;

struct DrivePoint : ScanPoint {
    static bool CheckWidthValid(const spScanPoint &pt1, const spScanPoint &pt2, float &width) {
        if (!pt1 || !pt2 || pt1->trafficline_ptr == pt2->trafficline_ptr) {
            return false;
        }
        float leftside = pt1->y;
        float shift = GetLineShift(pt1->line_type);
        leftside = leftside - shift;
        float rightside = pt2->y;
        shift = GetLineShift(pt2->line_type);
        rightside = rightside + shift;
        width = fabsf(leftside - rightside);
        if (width >= LANEWIDTH_MIN) {
            return true;
        }
        return false;
    }
    static float GetFrontPtAverageWidth(const std::shared_ptr<DrivePoint> this_thindrivept);
    DrivePoint()
        : ScanPoint(),
          is_closed(false),
          is_changelane(false),
          transition(LaneTransition::LaneTransition_Unknown),
          meetboundry(0),
          meet_boundry(SELECTLIKE::MIDDLE),
          meet_collision(false),
          meet_shadearea(SELECTLIKE::MIDDLE),
          meet_merge(false),
          do_shift(SELECTLIKE::MIDDLE),
          sub_left(),
          sub_right(),
          width(0.f),
          origin_width(width) {}

    virtual ~DrivePoint() {}

    void SetWidth(const float refinewidth) {
        width = refinewidth;
        origin_width = width;
    }

    void SetOriginWidthByPrevPt();

    float GetWidth() const { return width; }

    float GetOriginWidth() const { return origin_width; }

    void ToPoint(DrivePathPoint<float> &pt);

    bool is_closed;
    bool is_changelane;
    LaneTransition transition;
    int meetboundry;
    SELECTLIKE meet_boundry;
    bool meet_collision;
    SELECTLIKE meet_shadearea;
    bool meet_merge;
    SELECTLIKE do_shift;
    // 左侧scan point（位于车道线上）
    std::weak_ptr<ScanPoint> sub_left;
    // 右侧scan point（位于车道线上）
    std::weak_ptr<ScanPoint> sub_right;

   private:
    float width;
    float origin_width;
};
using spDrivePoint = std::shared_ptr<DrivePoint>;