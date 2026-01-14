import pygame
import math


class VehicleVisualizer:
    def __init__(
        self,
        screen_size=(1000, 800),
        bg_color=(30, 30, 30),
    ):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Vehicle Kinematic Model Visualization")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)

        self.W, self.H = screen_size

        # ===== 几何硬约束 =====
        # 显示范围：[-30, +30] m
        self.view_half_range = 80.0
        self.scale = self.W / (2.0 * self.view_half_range)

        self.bg_color = bg_color

        # 屏幕中心
        self.origin_x = self.W // 2
        self.origin_y = self.H // 2

        # 相机（世界坐标）——始终等于自车
        self.camera_x = 0.0
        self.camera_y = 0.0

        # 网格 / 刻度
        self.grid_world = 10.0
        self.tick_world = 10.0

        # 颜色
        self.vehicle_color = (50, 200, 255)
        self.obstacle_color = (255, 180, 60)
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
    def draw_grid(self):
        world_left = self.camera_x - self.view_half_range
        world_right = self.camera_x + self.view_half_range
        world_bottom = self.camera_y - self.view_half_range
        world_top = self.camera_y + self.view_half_range

        x = math.floor(world_left / self.grid_world) * self.grid_world
        while x <= world_right:
            sx1, sy1 = self.world_to_screen(x, world_bottom)
            sx2, sy2 = self.world_to_screen(x, world_top)
            pygame.draw.line(self.screen, self.grid_color, (sx1, sy1), (sx2, sy2), 1)
            x += self.grid_world

        y = math.floor(world_bottom / self.grid_world) * self.grid_world
        while y <= world_top:
            sx1, sy1 = self.world_to_screen(world_left, y)
            sx2, sy2 = self.world_to_screen(world_right, y)
            pygame.draw.line(self.screen, self.grid_color, (sx1, sy1), (sx2, sy2), 1)
            y += self.grid_world

    # ===============================
    # 世界坐标轴（x=0, y=0）
    # ===============================
    def draw_frame_axes(self):
        # ========= 世界原点在屏幕中的位置 =========
        sx0, sy0 = self.world_to_screen(0.0, 0.0)

        # ---------- Y 轴（x = 0） ----------
        if 0 <= sx0 <= self.W:
            x_axis_x = sx0
        elif sx0 < 0:
            x_axis_x = 0
        else:
            x_axis_x = self.W

        pygame.draw.line(
            self.screen,
            self.axis_color,
            (x_axis_x, 0),
            (x_axis_x, self.H),
            2,
        )

        # ---------- X 轴（y = 0） ----------
        if 0 <= sy0 <= self.H:
            y_axis_y = sy0
        elif sy0 < 0:
            y_axis_y = 0
        else:
            y_axis_y = self.H

        pygame.draw.line(
            self.screen,
            self.axis_color,
            (0, y_axis_y),
            (self.W, y_axis_y),
            2,
        )

        # ========= 刻度（世界坐标） =========
        world_left = self.camera_x - self.view_half_range
        world_right = self.camera_x + self.view_half_range
        world_bottom = self.camera_y - self.view_half_range
        world_top = self.camera_y + self.view_half_range

        # X 轴刻度
        x = math.ceil(world_left / self.tick_world) * self.tick_world
        while x <= world_right:
            sx, sy = self.world_to_screen(x, 0.0)
            pygame.draw.line(
                self.screen, self.axis_color, (sx, y_axis_y - 5), (sx, y_axis_y + 5), 1
            )
            label = self.font.render(f"{x:.0f}", True, self.text_color)
            self.screen.blit(label, (sx - 10, y_axis_y + 8))
            x += self.tick_world

        # Y 轴刻度
        y = math.ceil(world_bottom / self.tick_world) * self.tick_world
        while y <= world_top:
            sx, sy = self.world_to_screen(0.0, y)
            pygame.draw.line(
                self.screen, self.axis_color, (x_axis_x - 5, sy), (x_axis_x + 5, sy), 1
            )
            label = self.font.render(f"{y:.0f}", True, self.text_color)
            self.screen.blit(label, (x_axis_x + 8, sy - 7))
            y += self.tick_world

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
    # 单步刷新
    # ===============================
    def run_step(self, ego_vehicle, trajectory=None, obstacles=None):
        if obstacles is None:
            obstacles = []
        if trajectory is None:
            trajectory = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # ===== 相机严格居中自车 =====
        self.camera_x = ego_vehicle.x
        self.camera_y = ego_vehicle.y

        self.screen.fill(self.bg_color)

        self.draw_grid()
        self.draw_frame_axes()

        for obs in obstacles:
            self.draw_obstacle(obs)

        if trajectory:
            self.draw_path([(p.x, p.y) for p in trajectory])

        self.draw_vehicle(ego_vehicle)

        pygame.display.flip()
