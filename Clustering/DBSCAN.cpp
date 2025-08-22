#include <iostream>
#include <vector>

struct Particle {
  std::string obj_id;
  float x = 0.0f;
  float y = 0.0f;
  float weight = 0.0f;
  std::vector<float> xn;                              // 多维数据
  int cluster = 0;                                    // 第几个簇
  PointType point_type = PointType::kpointType_UNDO;  // 1 noise 2 border 3 core
  ParticleType particle_type =
      ParticleType::kparticleType_UNKNOWN;  // particle语义类型
  int pts = 0;                              // points in MinPts
  int corePointID = -1;                     // 核心点的标号
  std::vector<int> corepts;
  int visited = 0;
  int cluster_idx = 0;  // 用来聚类logical cluster时记录idx
  spTrafficLightInstance tfl_instance;
  Particle() = default;

  explicit Particle(std::string id, float x, float y, float weight,
                    ParticleType type)
      : obj_id(id), x(x), y(y), weight(weight), particle_type(type) {}
  void CalculateWeight(const Vec2D<float> &scan_start_points,
                       float scan_radius);
  void SetParticleType(const int &type);
};



void DBSCAN(std::vector<Particle> &dataset, keyVal keyValType, const float &Eps,
            const int &MinPts) {
  int clusterID = 0;
  size_t len = dataset.size();
  if (len == 0U) {
    return;
  }
  static std::map<keyVal,
                  std::function<float(const Particle &, const Particle &)>>
      distFuncMap = {{keyVal::keyVal_CENTER, squareDistance},
                     {keyVal::keyVal_THETA, thetaDistance},
                     {keyVal::keyVal_RADIUS, radiusDistance},
                     {keyVal::keyVal_X, CalcuXDistance},
                     {keyVal::keyVal_Y, CalcuYDistance}};
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
    SLOG_E << "Invalid keyValType! ";
  }

  size_t numCorePoint = corePoint.size();
  for (size_t i = 0U; i < numCorePoint; i++) {
    std::vector<float> &dist_i = distP2P[corePoint[i].corePointID];
    std::vector<int> &corepts_i = corePoint[i].corepts;
    for (int j = 0; static_cast<size_t>(j) < numCorePoint; j++) {
      float distTemp = dist_i[corePoint[j].corePointID];
      if (distTemp <= Eps) {
        // other point orderID link to core point
        corepts_i.push_back(j);
      }
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


void GetCluster(std::vector<Particle> &data,
                std::vector<std::vector<Particle>> &clusters) {
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