#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <stack>
#include <unordered_map>
#include <vector>

enum PointType : int {
    kpointType_UNDO = 0,
    kpointType_CORE = 3,
    kpointType_BORDER = 2,
    kpointType_NOISE = 1
};

enum class ParticleType : int {
    kparticleType_UNKNOWN = 0,         // 未分类
    kparticleType_StopLine = 1,        // 停止线
    kparticleType_CrossWalk = 2,       // 人行横道
    kparticleType_LaneLine = 3,        // 车道线
    kparticleType_RoadEdge = 4,        // 路沿
    kparticleType_Junction = 5,        // bev_junction
    kparticleType_TrafficLight = 6,    // 交通灯
    kparticleType_CarStreamPoint = 7,  // 专线点
    kparticleType_OutLine = 8,         // 轮廓线
};

enum class keyVal : int {
    keyVal_THETA = 0,       // 向量夹角，正反向会区分，范围0-pi
    keyVal_THETA_HALF = 5,  // 直线夹角，正反向不区分，范围0-pi/2
    keyVal_X = 1,           // x坐标
    keyVal_Y = 2,           // y坐标
    keyVal_CENTER = 3,      // 欧式距离
    keyVal_ALL = 4,
    keyVal_LATERAL = 6,        // 计算两条线之间的横向距离
    keyVal_CLUSTER = 7,        // 计算两个cluster之间的距离，依据line之间的距离
    keyVal_DISCRETECOMPS = 8,  // 计算两个离散要素(只支持人行横道和停止线)的距离
    keyVal_MIN_X_Y = 9,
    keyVal_RADIUS = 10,
};

struct Vec2D {
    float x;
    float y;
    Vec2D() = default;
    explicit Vec2D(float x, float y) : x(x), y(y) {}
};

struct Particle {
    std::string obj_id;
    float x = 0.0f;
    float y = 0.0f;
    float weight = 0.0f;
    std::vector<float> xn;                                             // 多维数据
    int cluster = 0;                                                   // 第几个簇
    PointType point_type = PointType::kpointType_UNDO;                 // 1 noise 2 border 3 core
    ParticleType particle_type = ParticleType::kparticleType_UNKNOWN;  // particle语义类型
    int pts = 0;                                                       // points in MinPts
    int corePointID = -1;                                              // 核心点的标号
    std::vector<int> corepts;
    int visited = 0;
    int cluster_idx = 0;  // 用来聚类logical cluster时记录idx
    // spTrafficLightInstance tfl_instance;
    Particle() = default;

    explicit Particle(std::string id, float x, float y, float weight, ParticleType type)
        : obj_id(id), x(x), y(y), weight(weight), particle_type(type) {}
    void CalculateWeight(const Vec2D &scan_start_points, float scan_radius);
    void SetParticleType(const int &type);
};

float squareDistance(const Particle &a, const Particle &b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void DBSCAN(std::vector<Particle> &dataset, keyVal keyValType, const float &Eps,
            const int &MinPts) {
    int clusterID = 0;
    size_t len = dataset.size();
    if (len == 0U) {
        return;
    }
    static std::map<keyVal, std::function<float(const Particle &, const Particle &)>> distFuncMap =
        {{keyVal::keyVal_CENTER, squareDistance}};
    auto iter = distFuncMap.find(keyValType);
    std::vector<std::vector<float>> distP2P(len, std::vector<float>(len, 0.0f));
    std::vector<Particle> corePoint;
    if (iter != distFuncMap.end()) {
        for (int i = 0; static_cast<size_t>(i) < len; i++) {
            // pts添加自己
            dataset[i].pts++;
            for (int j = i + 1; static_cast<size_t>(j) < len; j++) {
                float distance = (iter->second)(dataset[i], dataset[j]);
                // distP2P是对称矩阵
                distP2P[i][j] = distance;
                distP2P[j][i] = distance;
                if (distance <= Eps) {
                    dataset[i].pts++;
                    dataset[j].pts++;
                }
            }
            // core Particle 核心点，pts大于minPts的时候，该点为核心点
            if (dataset[i].pts >= MinPts) {
                dataset[i].point_type = PointType::kpointType_CORE;
                dataset[i].corePointID = i;
                corePoint.push_back(dataset[i]);
            }
        }
    } else {
        std::cout << "Invalid keyValType! " << std::endl;
    }
    // 打印出核心点
    size_t numCorePoint = corePoint.size();
    std::cout << "corePint ";
    for (auto &core : corePoint) {
        std::cout << core.obj_id << " ";
    }
    std::cout << std::endl;
    // 打印完毕

    for (size_t i = 0U; i < numCorePoint; i++) {
        std::vector<float> &dist_i = distP2P[corePoint[i].corePointID];
        std::vector<int> &corepts_i = corePoint[i].corepts;
        for (int j = 0; static_cast<size_t>(j) < numCorePoint; j++) {
            float distTemp = dist_i[corePoint[j].corePointID];
            std::cout << "CorePoint " << corePoint[i].obj_id << " corepts: ";
            if (distTemp <= Eps) {
                // other point orderID link to core point
                std::cout << corePoint[j].obj_id << " ";
                corepts_i.push_back(j);
            }
            std::cout << std::endl;
            // 把每一个在核心点领域的核心点放到一起
        }
    }
    for (size_t i = 0U; i < numCorePoint; i++) {
        // 遍历所有的核心点
        std::stack<Particle *> ps;
        if (corePoint[i].visited == 1) {
            continue;
        }
        clusterID++;
        corePoint[i].cluster = clusterID;
        ps.push(&corePoint[i]);
        Particle *v;
        while (!ps.empty()) {
            v = ps.top();
            v->visited = 1;
            ps.pop();
            for (size_t j = 0U; j < v->corepts.size(); j++) {
                // 最开始归类的一簇进行遍历
                if (corePoint[v->corepts[j]].visited == 1) {
                    continue;
                }
                corePoint[v->corepts[j]].cluster = corePoint[i].cluster;
                corePoint[v->corepts[j]].visited = 1;
                ps.push(&corePoint[v->corepts[j]]);
            }
        }
    }

    // border point,joint border point to core point
    // k用来在dataset中统计是第几个核心点
    int k = 0;
    for (size_t i = 0U; i < len; i++) {
        if (dataset[i].point_type == PointType::kpointType_CORE) {
            // 如果该点是核心点，在上面已经访问过了，就不再访问，
            // 因为核心点不可能是边界点，没必要再访问一次
            dataset[i].cluster = corePoint[k++].cluster;
            // 遍历到第k个核心点时，把属于的簇id给原来的dataset
            continue;
        }
        for (size_t j = 0U; j < numCorePoint; j++) {
            float distTemp = distP2P[i][corePoint[j].corePointID];
            if (distTemp <= Eps) {
                dataset[i].point_type = PointType::kpointType_BORDER;
                dataset[i].cluster = corePoint[j].cluster;
                break;
            }
        }
    }
}

void GetCluster(std::vector<Particle> &data, std::vector<std::vector<Particle>> &clusters) {
    std::unordered_map<int, std::vector<int>> cluster_ids;
    for (int i = 0; static_cast<size_t>(i) < data.size(); ++i) {
        cluster_ids[data[i].cluster].push_back(i);
    }

    for (auto &iter : cluster_ids) {
        std::vector<int> index = iter.second;
        std::vector<Particle> cluster;
        for (size_t i = 0U; i < index.size(); ++i) {
            cluster.push_back(data[index[i]]);
        }
        clusters.push_back(cluster);
    }
    return;
}

int main() {
    std::vector<Particle> dataset;
    std::vector<std::vector<Particle>> clusters;
    Particle p1("1", 1.0f, 1.0f, 1.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p2("2", 1.1f, 1.2f, 2.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p3("3", 1.5f, 1.3f, 3.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p4("4", 4.0f, 4.0f, 4.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p5("5", 5.0f, 5.0f, 5.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p6("6", 6.0f, 6.0f, 6.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p7("7", 7.0f, 7.0f, 7.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p8("8", 8.0f, 8.0f, 8.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p9("9", 9.0f, 9.0f, 9.0f, ParticleType::kparticleType_CarStreamPoint);
    Particle p10("10", 10.0f, 10.0f, 10.0f, ParticleType::kparticleType_CarStreamPoint);
    dataset.push_back(p1);
    dataset.push_back(p2);
    dataset.push_back(p3);
    dataset.push_back(p4);
    dataset.push_back(p5);
    dataset.push_back(p6);
    dataset.push_back(p7);
    dataset.push_back(p8);
    dataset.push_back(p9);
    dataset.push_back(p10);
    std::cout << "dataset size: " << dataset.size() << std::endl;
    DBSCAN(dataset, keyVal::keyVal_CENTER, 0.5f, 2);
    GetCluster(dataset, clusters);
    std::cout << "clusters size: " << clusters.size() << std::endl;
    for (size_t i = 0U; i < clusters.size(); i++) {
        std::cout << "cluster " << i << " size: " << clusters[i].size() << std::endl;
        for (size_t j = 0U; j < clusters[i].size(); j++) {
            std::cout << clusters[i][j].obj_id << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}