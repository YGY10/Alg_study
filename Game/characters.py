from dataclasses import dataclass
from typing import List, Optional


from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Character:
    # ===== 必填字段（无默认值一定要放前面）=====
    name: str
    rarity: str
    prob: float  # 抽卡权重，不是 0~1 概率

    # Lv1 基础属性
    base_force: float  # 武力
    base_intelligence: float  # 智力
    base_defence: float  # 防御
    base_speed: float  # 速度

    # 每级成长
    g_force: float
    g_intelligence: float
    g_defence: float
    g_speed: float

    # ===== 可选字段（有默认值放后面）=====
    portrait: Optional[str] = None
    skills: Optional[List[str]] = None


class CharacterInstance:
    """
    运行时实例：拥有等级 / 经验 / 星级 / 碎片等状态
    """

    def __init__(self, base: Character):
        self.base = base
        self.level = 1
        self.exp = 0.0

        self.star = 1  # 星级，从 1 星开始
        self.fragments = 0  # 碎片数量

    # ---------- 数值计算 ----------
    def _calc_attr(self, base, growth) -> float:
        """
        基础公式： (Lv1 属性 + (level - 1) * 成长) * 星级加成
        星级加成示例：每升 1 星，全属性 +10%
        """
        value = base + (self.level - 1) * growth
        star_multiplier = 1.0 + 0.1 * (self.star - 1)
        return value * star_multiplier

    @property
    def force(self) -> float:
        return self._calc_attr(self.base.base_force, self.base.g_force)

    @property
    def intelligence(self) -> float:
        return self._calc_attr(self.base.base_intelligence, self.base.g_intelligence)

    @property
    def defence(self) -> float:
        return self._calc_attr(self.base.base_defence, self.base.g_defence)

    @property
    def speed(self) -> float:
        return self._calc_attr(self.base.base_speed, self.base.g_speed)

    # ---------- 升级 / 碎片 ----------
    def exp_to_next(self) -> float:
        # 简单经验曲线：100 * level^1.2
        return 100.0 * (self.level**1.2)

    def add_exp(self, amount: float):
        self.exp += amount
        # 支持一次获得多级
        while self.exp >= self.exp_to_next():
            self.exp -= self.exp_to_next()
            self.level += 1

    def fragments_needed_for_next_star(self) -> int:
        # 示例：升星需要 star * 10 个碎片
        return self.star * 10

    def add_fragment(self, amount: int = 1):
        self.fragments += amount
        # 支持一次获得多星
        while self.fragments >= self.fragments_needed_for_next_star():
            self.fragments -= self.fragments_needed_for_next_star()
            self.star += 1

    # ---------- 存档 ----------
    def to_dict(self) -> dict:
        return {
            "name": self.base.name,
            "level": self.level,
            "exp": self.exp,
            "star": self.star,
            "fragments": self.fragments,
        }

    @staticmethod
    def from_dict(data: dict, base_char: Character) -> "CharacterInstance":
        inst = CharacterInstance(base_char)
        inst.level = data.get("level", 1)
        inst.exp = data.get("exp", 0.0)
        inst.star = data.get("star", 1)
        inst.fragments = data.get("fragments", 0)
        return inst


# =========================
# 武将数据
# =========================
CHARACTERS = [
    Character(
        name="刘备",
        rarity="SSR",
        prob=0.15,
        portrait="LiuBei.png",
        skills=["昭烈皇汉"],
        base_force=70.0,
        base_intelligence=80.0,
        base_defence=85.0,
        base_speed=50.0,
        g_force=1.5,
        g_intelligence=1.6,
        g_defence=1.8,
        g_speed=1.0,
    ),
    Character(
        name="关羽",
        rarity="SSR",
        prob=0.05,
        portrait="Guanyu.png",
        skills=["威震华夏"],
        base_force=98.0,
        base_intelligence=80.0,
        base_defence=80.0,
        base_speed=70.0,
        g_force=1.8,
        g_intelligence=1.0,
        g_defence=1.2,
        g_speed=1.2,
    ),
]

# =========================
# 稀有度颜色
# =========================
RARITY_COLOR = {
    "SSR": (255, 215, 0),
    "SR": (100, 200, 255),
    "R": (180, 180, 180),
}
