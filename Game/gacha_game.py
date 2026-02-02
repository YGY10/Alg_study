import os
import json
import random
from typing import List, Tuple

import pygame
from characters import CHARACTERS, CharacterInstance, RARITY_COLOR

# =========================
# 初始化
# =========================
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gacha Game")

FONT_PATH = os.path.join("asserts", "fonts", "NotoSansCJK-Regular.ttc")
font = pygame.font.Font(FONT_PATH, 22)
big_font = pygame.font.Font(FONT_PATH, 32)

clock = pygame.time.Clock()

ASSET_DIR = os.path.join("asserts", "portraits")
SAVE_PATH = "save.json"

# =========================
# UI 状态
# =========================
UI_GACHA = "gacha"
UI_RESULT = "result"
UI_INVENTORY = "inventory"
UI_DETAIL = "detail"

# =========================
# 立绘加载
# =========================
PORTRAITS = {}
for c in CHARACTERS:
    if c.portrait:
        path = os.path.join(ASSET_DIR, c.portrait)
        if os.path.exists(path):
            PORTRAITS[c.name] = pygame.image.load(path).convert_alpha()


# =========================
# 玩家仓库
# =========================
class PlayerInventory:
    def __init__(self):
        self.characters: List[CharacterInstance] = []

    def find(self, name: str):
        for c in self.characters:
            if c.base.name == name:
                return c
        return None

    def add(self, inst: CharacterInstance):
        existing = self.find(inst.base.name)
        if existing:
            existing.add_fragment()
            return "fragment", existing
        self.characters.append(inst)
        return "new", inst

    def to_dict(self):
        return {"characters": [c.to_dict() for c in self.characters]}

    def save(self):
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load():
        inv = PlayerInventory()
        if not os.path.exists(SAVE_PATH):
            return inv
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if not txt:
                    return inv
                data = json.loads(txt)
        except Exception:
            return inv

        for d in data.get("characters", []):
            base = next((c for c in CHARACTERS if c.name == d["name"]), None)
            if base:
                inv.characters.append(CharacterInstance.from_dict(d, base))
        return inv


# =========================
# 抽卡
# =========================
def gacha_draw():
    r = random.random()
    acc = 0.0
    for c in CHARACTERS:
        acc += c.prob
        if r <= acc:
            return CharacterInstance(c)
    return CharacterInstance(CHARACTERS[-1])


# =========================
# UI 工具
# =========================
def draw_button(rect, text):
    pygame.draw.rect(screen, (60, 60, 60), rect)
    pygame.draw.rect(screen, (200, 200, 200), rect, 2)
    t = font.render(text, True, (255, 255, 255))
    screen.blit(t, t.get_rect(center=rect.center))


def draw_portrait(inst, cx, top):
    if not inst or inst.base.name not in PORTRAITS:
        return
    img = PORTRAITS[inst.base.name]
    iw, ih = img.get_size()
    scale = min(320 / iw, 480 / ih)
    img = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
    screen.blit(img, (cx - img.get_width() // 2, top))


# =========================
# 主循环
# =========================
def main():
    inventory = PlayerInventory.load()

    ui_state = UI_GACHA
    last_result = None
    last_result_type = ""
    selected = None

    # 按钮
    btn_home = pygame.Rect(20, 20, 120, 40)
    btn_draw = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 120, 200, 50)
    btn_continue = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 60, 200, 40)
    btn_inventory = pygame.Rect(WIDTH - 170, 20, 150, 40)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                inventory.save()
                running = False

            elif e.type == pygame.MOUSEBUTTONDOWN:
                # ===== 全局返回首页（最高优先级）=====
                if btn_home.collidepoint(e.pos):
                    ui_state = UI_GACHA
                    selected = None
                    continue

                # ===== 右上角角色列表 =====
                if btn_inventory.collidepoint(e.pos):
                    ui_state = UI_INVENTORY
                    continue

                # ===== 抽卡 =====
                if ui_state == UI_GACHA and btn_draw.collidepoint(e.pos):
                    inst = gacha_draw()
                    last_result_type, last_result = inventory.add(inst)
                    inventory.save()
                    ui_state = UI_RESULT

                # ===== 继续抽卡（真正再抽一次）=====
                elif ui_state == UI_RESULT and btn_continue.collidepoint(e.pos):
                    inst = gacha_draw()
                    last_result_type, last_result = inventory.add(inst)
                    inventory.save()
                    ui_state = UI_RESULT

                # ===== 仓库点击 =====
                elif ui_state == UI_INVENTORY:
                    y = 100
                    for inst in inventory.characters:
                        rect = pygame.Rect(40, y, 400, 32)
                        if rect.collidepoint(e.pos):
                            selected = inst
                            ui_state = UI_DETAIL
                            break
                        y += 40

        # =========================
        # 绘制
        # =========================
        screen.fill((25, 25, 25))
        draw_button(btn_home, "返回首页")
        draw_button(btn_inventory, "角色列表")

        if ui_state == UI_GACHA:
            t = big_font.render("抽卡界面", True, (255, 255, 255))
            screen.blit(t, (WIDTH // 2 - t.get_width() // 2, 60))
            draw_button(btn_draw, "抽 卡")

        elif ui_state == UI_RESULT and last_result:
            t = big_font.render("抽卡结果", True, (255, 255, 255))
            screen.blit(t, (WIDTH // 2 - t.get_width() // 2, 60))

            name = big_font.render(
                last_result.base.name,
                True,
                RARITY_COLOR[last_result.base.rarity],
            )
            screen.blit(name, (WIDTH // 2 - name.get_width() // 2, 110))

            msg = (
                "获得新武将！"
                if last_result_type == "new"
                else f"已拥有 → 碎片 {last_result.fragments}"
            )
            m = font.render(msg, True, (220, 220, 220))
            screen.blit(m, (WIDTH // 2 - m.get_width() // 2, 150))

            draw_portrait(last_result, WIDTH // 2, 180)
            draw_button(btn_continue, "继续抽卡")

        elif ui_state == UI_INVENTORY:
            t = big_font.render("角色列表", True, (255, 255, 255))
            screen.blit(t, (40, 60))
            y = 100
            for inst in inventory.characters:
                txt = font.render(
                    f"{inst.base.name}  Lv.{inst.level}  ⭐{inst.star}  碎片:{inst.fragments}",
                    True,
                    RARITY_COLOR[inst.base.rarity],
                )
                screen.blit(txt, (40, y))
                y += 40

        elif ui_state == UI_DETAIL and selected:
            t = big_font.render(
                selected.base.name, True, RARITY_COLOR[selected.base.rarity]
            )
            screen.blit(t, (40, 60))
            draw_portrait(selected, 250, 120)

            y = 120
            for line in [
                f"等级: {selected.level}",
                f"武力: {selected.force:.1f}",
                f"智力: {selected.intelligence:.1f}",
                f"防御: {selected.defence:.1f}",
                f"速度: {selected.speed:.1f}",
                f"碎片: {selected.fragments}",
            ]:
                screen.blit(font.render(line, True, (220, 220, 220)), (500, y))
                y += 32

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
