import pygame
import math


class VehicleVisualizer:
    def __init__(
        self,
        screen_size=(1000, 800),
        scale=20.0,  # 1 m = 20 px
        bg_color=(30, 30, 30),
    ):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Vehicle Kinematic Model Visualization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)

        self.W, self.H = screen_size
        self.scale = scale
        self.bg_color = bg_color

        # 屏幕中心（世界视窗中心）
        self.origin_x = self.W // 2
        self.origin_y = self.H // 2

        # 相机（世界坐标）
        self.camera_x = 0.0
        self.camera_y = 0.0

        # 颜色
        self.vehicle_color = (50, 200, 255)  # 自车
        self.obstacle_color = (255, 180, 60)  # 障碍物
        self.heading_color = (255, 80, 80)
        self.axis_color = (200, 200, 200)
        self.grid_color = (70, 70, 70)
        self.text_color = (220, 220, 220)

    # ===============================
    # 世界 → 屏幕
    # ===============================
    def world_to_screen(self, x, y):
        sx = int((x - self.camera_x) * self.scale + self.origin_x)
        sy = int(self.origin_y - (y - self.camera_y) * self.scale)
        return sx, sy

    # ===============================
    # 世界网格
    # ===============================
    def draw_grid(self, grid_world=5.0):
        world_left = self.camera_x - self.origin_x / self.scale
        world_right = self.camera_x + self.origin_x / self.scale
        world_bottom = self.camera_y - self.origin_y / self.scale
        world_top = self.camera_y + self.origin_y / self.scale

        x = math.floor(world_left / grid_world) * grid_world
        while x <= world_right:
            sx1, sy1 = self.world_to_screen(x, world_bottom)
            sx2, sy2 = self.world_to_screen(x, world_top)
            pygame.draw.line(self.screen, self.grid_color, (sx1, sy1), (sx2, sy2), 1)
            x += grid_world

        y = math.floor(world_bottom / grid_world) * grid_world
        while y <= world_top:
            sx1, sy1 = self.world_to_screen(world_left, y)
            sx2, sy2 = self.world_to_screen(world_right, y)
            pygame.draw.line(self.screen, self.grid_color, (sx1, sy1), (sx2, sy2), 1)
            y += grid_world

    # ===============================
    # 图像框坐标轴（左 / 下）
    # ===============================
    def draw_frame_axes(self, tick_world=5.0):
        world_left = self.camera_x - self.origin_x / self.scale
        world_right = self.camera_x + self.origin_x / self.scale
        world_bottom = self.camera_y - self.origin_y / self.scale
        world_top = self.camera_y + self.origin_y / self.scale

        # Y 轴（左）
        pygame.draw.line(self.screen, self.axis_color, (0, 0), (0, self.H), 2)
        # X 轴（下）
        pygame.draw.line(
            self.screen, self.axis_color, (0, self.H - 1), (self.W, self.H - 1), 2
        )

        # X 轴刻度
        x = math.ceil(world_left / tick_world) * tick_world
        while x <= world_right:
            sx, _ = self.world_to_screen(x, world_bottom)
            pygame.draw.line(
                self.screen, self.axis_color, (sx, self.H - 6), (sx, self.H), 1
            )
            label = self.font.render(f"{x:.1f}", True, self.text_color)
            self.screen.blit(label, (sx - 12, self.H - 22))
            x += tick_world

        # Y 轴刻度
        y = math.ceil(world_bottom / tick_world) * tick_world
        while y <= world_top:
            _, sy = self.world_to_screen(world_left, y)
            pygame.draw.line(self.screen, self.axis_color, (0, sy), (6, sy), 1)
            label = self.font.render(f"{y:.1f}", True, self.text_color)
            self.screen.blit(label, (8, sy - 7))
            y += tick_world

    # ===============================
    # 画自车
    # ===============================
    def draw_vehicle(self, vehicle):
        corners = vehicle.corners()
        pts = [self.world_to_screen(x, y) for x, y in corners]
        pygame.draw.polygon(self.screen, self.vehicle_color, pts, 2)

        x, y, yaw, _ = vehicle.state()
        start = self.world_to_screen(x, y)
        end = self.world_to_screen(
            x + vehicle.length * 0.6 * math.cos(yaw),
            y + vehicle.length * 0.6 * math.sin(yaw),
        )
        pygame.draw.line(self.screen, self.heading_color, start, end, 3)

    # ===============================
    # 画障碍物车辆
    # ===============================
    def draw_obstacle(self, vehicle):
        corners = vehicle.corners()
        pts = [self.world_to_screen(x, y) for x, y in corners]
        pygame.draw.polygon(self.screen, self.obstacle_color, pts, 2)

    def draw_path(self, path, color=(0, 255, 0)):
        if len(path) < 2:
            return
        pts = [self.world_to_screen(x, y) for x, y in path]
        pygame.draw.lines(self.screen, color, False, pts, 2)

    # ===============================
    # 主循环（支持多车）
    # ===============================
    def run_step(self, ego_vehicle, trajectory=None, obstacles=None):
        if obstacles is None:
            obstacles = []
        if trajectory is None:
            trajectory = []

        # -------- 必须：处理事件 --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 相机跟随自车
        self.camera_x = ego_vehicle.x
        self.camera_y = ego_vehicle.y

        self.screen.fill(self.bg_color)

        # 背景
        self.draw_grid(grid_world=5.0)
        self.draw_frame_axes(tick_world=5.0)

        # 障碍物
        for obs in obstacles:
            self.draw_obstacle(obs)

        # 轨迹
        if trajectory:
            self.draw_path([(p.x, p.y) for p in trajectory])

        # 自车
        self.draw_vehicle(ego_vehicle)

        # -------- 必须：刷新屏幕 --------
        pygame.display.flip()

    def run(
        self,
        ego_vehicle,
        obstacles=None,
        update_callback=None,
        path_provider=None,
        fps=60,
    ):
        if obstacles is None:
            obstacles = []

        running = True
        while running:
            dt = self.clock.tick(fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 更新自车（控制器）
            if update_callback:
                update_callback(ego_vehicle, dt)

            # 相机只跟随自车
            self.camera_x = ego_vehicle.x
            self.camera_y = ego_vehicle.y

            self.screen.fill(self.bg_color)

            # 背景
            self.draw_grid(grid_world=5.0)
            self.draw_frame_axes(tick_world=5.0)

            # 障碍物
            for obs in obstacles:
                self.draw_obstacle(obs)

            # ===== 关键新增：画规划路径 =====
            if path_provider:
                path = path_provider(ego_vehicle, obstacles)
                if path:
                    self.draw_path(path)

            # 自车
            self.draw_vehicle(ego_vehicle)

            pygame.display.flip()

        pygame.quit()
