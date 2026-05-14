class BattleLog:
    def __init__(self):
        self.entries = []

    def add(self, message: str) -> None:
        self.entries.append(message)

    def print(self) -> None:
        for entry in self.entries:
            print(entry)

    def __iter__(self):
        return iter(self.entries)

    def __str__(self):
        return "\n".join(self.entries)
