# config/bev_config.py

# BEV coordinate is ego vehicle local coordinate:
# x forward
# y right
# z up
#
# BEV image convention:
# row=0       -> x = BEV_X_MAX
# row=height  -> x = BEV_X_MIN
# col=0       -> y = BEV_Y_MIN
# col=width   -> y = BEV_Y_MAX

# 至少覆盖前方 80m
BEV_X_MIN = -20.0
BEV_X_MAX = 80.0

BEV_Y_MIN = -30.0
BEV_Y_MAX = 30.0

# 先用 0.2 验证 8 camera 几何和融合
# 后面如果性能可以，再改成 0.1
BEV_RESOLUTION = 0.2

BEV_BACKGROUND_COLOR = (0, 0, 0)