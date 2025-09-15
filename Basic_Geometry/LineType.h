// LineType，almost same as line.v2.proto
enum LineType : uint64_t {
    LINE_TYPE_UNKNOWN = 0,             // 0 << 0
    LINE_TYPE_LANELINE = 2,            // 1 << 1
    LINE_TYPE_CURB = 4,                // 1 << 2
    LINE_TYPE_CENTER = 8,              // 1 << 3, Center line,
                                       // virtual and in the median of lane
    LINE_TYPE_GUARDRAIL = 16,          // 1 << 4
    LINE_TYPE_CONCRETEBARRIER = 32,    // 1 << 5
    LINE_TYPE_FENCE = 64,              // 1 << 6
    LINE_TYPE_WALL = 128,              // 1 << 7
    LINE_TYPE_CANOPY = 256,            // 1 << 8
    LINE_TYPE_VIRTUAL = 512,           // 1 << 9
    LINE_TYPE_CONE = 1024,             // 1 << 10
    LINE_TYPE_WATERHORSE = 2048,       // 1 << 11
    LINE_TYPE_GROUNDSIDE = 4096,       // 1 << 12
    LINE_TYPE_ROADEDGEUNKNOWN = 8192,  // 1 << 13
};