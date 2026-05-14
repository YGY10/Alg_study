class TargetSelector:
    @staticmethod
    def enemies_of(battle, hero):
        return battle.team_b if hero in battle.team_a else battle.team_a

    @staticmethod
    def allies_of(battle, hero):
        return battle.team_a if hero in battle.team_a else battle.team_b

    @staticmethod
    def alive(units):
        return [u for u in units if u.alive]

    @classmethod
    def random_enemy(cls, battle, hero):
        targets = cls.alive(cls.enemies_of(battle, hero))
        return battle.rng.choice(targets) if targets else None

    @classmethod
    def lowest_hp_enemy(cls, battle, hero):
        targets = cls.alive(cls.enemies_of(battle, hero))
        return min(targets, key=lambda h: h.hp) if targets else None

    @classmethod
    def all_enemies(cls, battle, hero):
        return cls.alive(cls.enemies_of(battle, hero))

    @classmethod
    def lowest_hp_ally(cls, battle, hero):
        targets = cls.alive(cls.allies_of(battle, hero))
        return min(targets, key=lambda h: h.hp) if targets else None

    @classmethod
    def all_allies(cls, battle, hero):
        return cls.alive(cls.allies_of(battle, hero))
