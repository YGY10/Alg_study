import random
import time
import os


class StockGame:
    def __init__(self):
        self.day = 1
        self.cash = 0.0
        self.init_cash = 0.0
        self.target_cash = 0.0
        self.is_paused = False

        # 初始化 10 只极具特色的随机股票 [名称, 当前价格, 波动属性(稳健/妖股/夕阳/成长)]
        self.stocks = {
            "000001 腾飞科技": {"price": 50.0, "type": "growth"},  # 高成长，暴涨暴跌
            "600002 宇宙茅台": {
                "price": 150.0,
                "type": "stable",
            },  # 绩优股，波动小但稳健
            "000003 黄金万两": {
                "price": 20.0,
                "type": "hedge",
            },  # 避险资产，逆势往往有惊喜
            "600004 远洋动力": {"price": 35.0, "type": "growth"},  # 周期股
            "000005 妖股王": {
                "price": 8.0,
                "type": "monster",
            },  # 绝对的妖股，可能翻倍也可能腰斩
            "600006 步步高升": {"price": 18.0, "type": "stable"},  # 慢牛股
            "000007 暴雷股份": {
                "price": 12.0,
                "type": "risk",
            },  # 高风险，随时可能退市跌停
            "600008 环球影业": {"price": 25.0, "type": "growth"},  # 正常波动
            "000009 传统燃油": {
                "price": 40.0,
                "type": "sunset",
            },  # 夕阳产业，阴跌概率大
            "600010 绿能生态": {
                "price": 15.0,
                "type": "growth",
            },  # 新能源，经常有政策利好
        }

        # 玩家持仓情况 {"股票名称": 持股数量}
        self.portfolio = {name: 0 for name in self.stocks}

    def start(self):
        print("=" * 45)
        print("        📈 欢迎来到大空头：模拟股市游戏 📉        ")
        print("=" * 45)
        while True:
            try:
                self.init_cash = float(input("请输入你的初始本金 (例如 100000): "))
                if self.init_cash <= 0:
                    raise ValueError
                break
            except ValueError:
                print("❌ 输入错误，请输入正整数本金！")

        self.cash = self.init_cash
        self.target_cash = self.init_cash * 3  # 设定通关上限：3倍本金

        print(f"\n🎯 游戏目标：当总资产达到 💰{self.target_cash:.2f} 元时通关！")
        print("⚠️ 失败条件：总资产归零或为负数时破产。")
        input("按【回车键】正式开盘...")

        self.game_loop()

    def update_stock_prices(self):
        """每一天股票价格的随机走向逻辑"""
        for name, info in self.stocks.items():
            current_price = info["price"]
            stock_type = info["type"]

            # 根据股票属性决定涨跌概率和幅度
            if stock_type == "stable":
                change_pct = random.uniform(-0.03, 0.04)  # -3% 到 +4%
            elif stock_type == "growth":
                change_pct = random.uniform(-0.06, 0.08)  # -6% 到 +8%
            elif stock_type == "monster":
                change_pct = random.uniform(-0.20, 0.25)  # 极高波动，类似2CM大长腿
            elif stock_type == "risk":
                change_pct = random.uniform(-0.15, 0.10)  # 跌易涨难
            elif stock_type == "sunset":
                change_pct = random.uniform(-0.05, 0.03)  # 缓慢阴跌
            else:
                change_pct = random.uniform(-0.05, 0.05)

            # 计算新价格，确保股票不会跌到0元以下
            new_price = max(0.1, current_price * (1 + change_pct))
            info["price"] = round(new_price, 2)

    def get_total_assets(self):
        """计算总资产 = 现金 + 所有持仓股票市值"""
        stock_value = sum(
            self.portfolio[name] * info["price"] for name, info in self.stocks.items()
        )
        return self.cash + stock_value

    def show_dashboard(self):
        """渲染每日股市行情盘面"""
        os.system("cls" if os.name == "nt" else "clear")  # 清屏，保持盘面干净
        total_assets = self.get_total_assets()

        print("=" * 60)
        print(
            f" 📅 虚拟交易日: 第 {self.day} 天  |  ⏸️ 暂停状态: {'已暂停' if self.is_paused else '运行中'}"
        )
        print(
            f" 💵 可用现金: {self.cash:.2f} 元 | 📊 当前总资产: {total_assets:.2f} 元"
        )
        print(
            f" 🏆 通关目标: {self.target_cash:.2f} 元 (进度: {(total_assets/self.target_cash)*100:.1f}%)"
        )
        print("=" * 60)
        print(f"{'股票代码及名称':<18}{'当前股价':<12}{'你的持仓':<10}{'当前市值':<12}")
        print("-" * 60)

        # 打印股票列表
        for idx, (name, info) in enumerate(self.stocks.items(), 1):
            holding = self.portfolio[name]
            market_value = holding * info["price"]
            print(
                f"[{idx}] {name:<12}\t{info['price']:<12.2f}{holding:<10}{market_value:<12.2f}"
            )
        print("=" * 60)

    def buy_stock(self):
        """买入股票逻辑"""
        try:
            idx = int(input("请输入要买入的股票编号 (1-10): ")) - 1
            name = list(self.stocks.keys())[idx]
            price = self.stocks[name]["price"]

            print(f"当前可用现金: {self.cash:.2f} 元，该股单价: {price:.2f} 元")
            amount = int(input("请输入买入股数 (整百数量): "))

            if amount <= 0:
                return
            cost = amount * price
            if cost > self.cash:
                print("❌ 余额不足，买不起这么多股！")
                time.sleep(1.5)
            else:
                self.cash -= cost
                self.portfolio[name] += amount
                print(f"🎉 成功买入 {name} {amount} 股！")
                time.sleep(1)
        except (ValueError, IndexError):
            print("❌ 输入编号或数量有误！")
            time.sleep(1.5)

    def sell_stock(self):
        """卖出股票逻辑"""
        try:
            idx = int(input("请输入要卖出的股票编号 (1-10): ")) - 1
            name = list(self.stocks.keys())[idx]
            price = self.stocks[name]["price"]
            holding = self.portfolio[name]

            if holding <= 0:
                print("❌ 你根本没有持仓这只股票！")
                time.sleep(1.5)
                return

            print(f"当前持有该股: {holding} 股，当前市价: {price:.2f} 元")
            amount = int(input(f"请输入卖出股数 (1-{holding}): "))

            if amount <= 0 or amount > holding:
                print("❌ 卖出数量不合法！")
                time.sleep(1.5)
            else:
                self.cash += amount * price
                self.portfolio[name] -= amount
                print(f"💰 成功卖出 {name} {amount} 股！")
                time.sleep(1)
        except (ValueError, IndexError):
            print("❌ 输入有误！")
            time.sleep(1.5)

    def game_loop(self):
        """游戏主循环"""
        while True:
            self.show_dashboard()
            total_assets = self.get_total_assets()

            # 1. 判断输赢条件
            if total_assets >= self.target_cash:
                print("\n🎉🎉🎉【恭喜通关】🎉🎉🎉")
                print(
                    f"你展现了股神般的华尔街操盘技术！最终总资产达到: {total_assets:.2f} 元！"
                )
                break
            if (
                total_assets <= 0 or self.cash < -1000
            ):  # 允许极小额度负数穿仓，大额负数算破产
                print("\n😭【游戏失败：你破产了】😭")
                print("强行平仓！你的资产已经亏光，被请出了交易大厅。")
                break

            # 2. 接收玩家操作指令
            print("💡 操作指南: [b]买入 | [s]卖出 | [n]下一天 | [p]暂停/恢复")
            action = input("请输入操作指令并回车: ").strip().lower()

            if action == "b":
                self.buy_stock()
            elif action == "s":
                self.sell_stock()
            elif action == "p":
                self.is_paused = not self.is_paused
                print("游戏状态已切换！")
                time.sleep(1)
            elif action == "n" or action == "":
                if self.is_paused:
                    print(
                        "⏸️ 游戏当前处于暂停状态，请先按 [p] 恢复时间流动再进入下一天。"
                    )
                    time.sleep(2)
                else:
                    # 时间流动，更新价格
                    self.update_stock_prices()
                    self.day += 1
            else:
                print("❌ 未知指令，请重新输入。")
                time.sleep(1)


if __name__ == "__main__":
    game = StockGame()
    game.start()
