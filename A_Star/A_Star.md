# A* 算法的流程
0. 两个优先队列：
   OpenSet(待处理的节点集合，即尚未完全探索的点), 
   ClosedSet(已经处理的节点集合，即不再需要访问的节点)。
   一个Map:
   用来标记已经处理过的所有结点，包括openset和close set的 all_node
1. 将起始节点加入到OpenSet, g(n) 初始化为0, f(n)初始化为h(start);
   初始化ClosedSet为空。
2. 当开放列表不为空时，进入主循环：
	当前节点为OpenSet中f(n)值最小的节点


