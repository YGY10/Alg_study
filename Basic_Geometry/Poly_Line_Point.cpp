#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
class Point {
   public:
    Point(const T& x, const T& y) : x_(x), y_(y) {}
    T DistanceTo(const Point<T>& other_point) const {
        return std::hypot(x_ - other_point.x_, y_ - other_point.y_);
    }
    const T& x() const { return x_; }
    const T& y() const { return y_; }

   private:
    T x_ = static_cast<T>(0.0);
    T y_ = static_cast<T>(0.0);
};

template <typename T>
class Line {
   public:
    Line(const Point<T>& start, const Point<T>& end)
        : start_(start), end_(end), unit_direction_(0, 0) {
        const T dx = end_.x() - start_.x();
        const T dy = end_.y() - start_.y();
        length_ = std::hypot(dx, dy);
        unit_direction_ = Point<T>(dx / length_, dy / length_);
        heading_ = std::atan2(dy, dx);
    }

    bool IsWithin(const T& value, T min, T max) const {
        if (min > max) std::swap(min, max);
        return value >= min - static_cast<T>(1e-6) && value <= max + static_cast<T>(1e-6);
    }
    bool IsPointIn(const Point<T>& other_point) const {
        if (length_ < static_cast<T>(1e-6)) {
            return std::abs(other_point.x() - start_.x()) < static_cast<T>(1e-6) &&
                   std::abs(other_point.y() - start_.y()) < static_cast<T>(1e-6);
        }
        Point<T> tmp_1 = start_ - other_point;
        Point<T> tmp_2 = end_ - other_point;
        T cross = tmp_1.x() * tmp_2.y() - tmp_1.y() * tmp_2.x();
        if (std::abs(cross) > static_cast<T>(1e-6)) {
            return false;
        }
        return IsWithin(other_point.x(), start_.x(), end_.x()) &&
               IsWithin(other_point.y(), start_.y(), end_.y());
    }
    T DistanceTo(const Point<T>& other_point) const {
        if (length_ < static_cast<T>(1e-6)) {
            return start_.DistanceTo(other_point);
        }
        const T x0 = other_point.x() - start_.x();
        const T y0 = other_point.y() - start_.y();
        const T proj = x0 * unit_direction_.x() + y0 * unit_direction_.y();
        if (proj < static_cast<T>(0.0)) {
            return std::hypot(x0, y0);
        }
        if (proj > length_) {
            return other_point.DistanceTo(end_);
        }
        return std::abs(x0 * unit_direction_.y() - y0 * unit_direction_.x());
    }

   private:
    Point<T> start_;
    Point<T> end_;
    Point<T> unit_direction_;
    T heading_;
    T length_;
};

template <typename T>
class Polygon {
   public:
    Polygon(const std::vector<Point<T>>& points) {
        points_.assign(points.begin(), points.end());
        num_points_ = static_cast<int>(points_.size());
        T cx = static_cast<T>(0.0);
        T cy = static_cast<T>(0.0);
        for (const auto& point : points_) {
            cx += point.x();
            cy += point.y();
        }
        cx /= static_cast<T>(num_points_) cy /= static_cast<T>(num_points_);
        center_ = Point<T>(cx, cy);
        area_ = static_cast<T>(0.0);
        for (int i = 1; i < num_points_; ++i) {
            Point<T> tmp_1 = points_[i] - points_[0];
            Point<T> tmp_2 = points_[i - 1] - points_[0];
            area_ += tmp_1.x() * tmp_2.y() - tmp_1.y() * tmp_2.x();
        }
        if (area_ < 0) {
            area_ = -area_;
            std::reverse(points_.begin(), points_.end());
        }
        area_ /= static_cast<T>(2.0);

        lines_.reserve(static_cast<size_t>(num_points_));
        for (int i = 0; i < num_points_; ++i) {
            Line<T> line(points_[i], points_[(i + 1) % num_points_]);
            lines_.emplace_back(line);
        }

        is_convex_ = true;
        for (int i = 0; i < num_points_; ++i) {
            Point<T> tmp_prev = points_[(i - 1 + num_points_) % num_points_] - points_[i];
            Point<T> tmp_curr = points_[(i + 1) % num_points_] - points_[i];
            T cross = tmp_prev.x() * tmp_curr.y() - tmp_prev.y() * tmp_curr.x();
            if (cross < -1e-6) {
                is_convex_ = false;
                break;
            }
        }
        //
        min_x_ = points_[0].x();
        max_x_ = points_[0].x();
        min_y_ = points_[0].y();
        max_y_ = points_[0].y();
        for (const auto& point : points_) {
            min_x_ = std::min(min_x_, point.x());
            max_x_ = std::max(max_x_, point.x());
            min_y_ = std::min(min_y_, point.y());
            max_y_ = std::max(max_y_, point.y());
        }
    }

    bool IsPointOnBoundary(const Point<T>& point) {
        return std::any_of(lines_.begin(), lines_.end(),
                           [&](const Line<T>& line) { return line.IsPointIn(point); })
    }

    bool IsPointIn(const Point<T>& point) {
        // 如果点在边界上，那么认为点在多边形内
        if (IsPointOnBoundary(point)) {
            return true;
        }
        int j = num_points_ - 1;
        int c = 0;  // 点和多边形的交点个数
        for (int i = 0; i < num_points_; ++i) {
            // 判断两个端点的y值的是否在point的两侧，如果在两侧，有可能在内侧，进一步判断
            // 如果不在，则不可能在现在遍历的这条边的内侧
            if ((points_[i].y() > point.y()) != (points_[j].y() > point.y())) {
                // 计算叉乘
                // 如果边的第一个点（p1）y小于边上的第二个点(p2)的y， 并且side > 0(叉乘的定义：pp1
                // -> pp2是逆时针的) 如果边的第二个点（p2）y小于边上的第一个点(p1)的y, 并且side <
                // 0(叉乘的定义：pp1 -> pp2是顺时针的)
                //
                Point<T> tmp_1 = points_[i] - point;
                Point<T> tmp_2 = points_[j] - point;
                const T side = tmp_1.x() * tmp_2.y() - tmp_1.y() * tmp_2.x();
                if (points_[i].y() < points_[j].y() ? side > static_cast<T>(0.0)
                                                    : side < static_cast<T>(0.0)) {
                    ++c;
                }
            }
            j = i;
        }
        return static_cast<bool>(c & 1);
    }
    T DistanceTo(const Point<T>& other_point) const {
        if (IsPointIn(point)) {
            return static_cast<T>(0.0);
        }
        T distance = std::numeric_limits<T>::infinity();
        for (int i = 0; i < num_points_; ++i) {
            distance = std::min(distance, lines_[i].DistanceTo(other_point));
        }
        return distance;
    }

   private:
    Point<T> center_;
    std::vector<Point<T>> points_;
    std::vector<Line<T>> lines_;
    int num_points_ = 0;
    bool is_convex_ = false;
    T area_ = static_cast<T>(0.0);
    T min_x_ = static_cast<T>(0.0);
    T max_x_ = static_cast<T>(0.0);
    T min_y_ = static_cast<T>(0.0);
    T max_y_ = static_cast<T>(0.0);
};

int main() {
    Point<double> p1(1.0, 0.0);
    Point<double> p2(0.0, 0.0);
    Point<double> p3(2.0, 2.0);
    Line<double> l1(p2, p3);
    std::cout << l1.DistanceTo(p1) << std::endl;
    Polygon<double> poly({p1, p2, p3});
    std::cout << poly.DistanceTo(p1) << std::endl;
    Point<double> p4(1.0, 1.0);
    std::cout << poly.IsPointIn(p4) << std::endl;
    return 0;
}
