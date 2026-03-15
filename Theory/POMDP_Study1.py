# Tiger Problem
# 面前有两扇门 左门left 右门right
# 老虎和宝藏各在一边 但人观察不到真实状态
# 人可以做三个动作： listen：听一下老虎在哪边 open_left:打开左门 open_right:打开右门
# 规则：
# 如果打开了有老虎的门，大奖励负值，比如 -100
# 如果打开了宝藏门，奖励正值，比如 +10
# 听一下也有代价，比如 -1
# 听到的信息不一定准，比如 85% 正确
import random

# 隐藏状态：老虎在哪边

STATES = ["tiger_left", "tiger_right"]
# 动作
ACTIONS = ["listen", "open_left", "open_right"]
# 观测
OBSERVATIONS = ["hear_left", "hear_right"]
# 参数
LISTEN_COST = -1
OPEN_TREASURE_REWARD = 10
OPEN_TIGER_PENALTY = -100
# 听的正确率
LISTEN_ACCURACY = 0.85


# 观测模型
def sample_observation(state, action):
    # 根据真实状态和动作，采样一个观测
    if action != "listen":
        return None
    if state == "tiger_left":
        return "hear_left" if random.random() < LISTEN_ACCURACY else "hear_right"
    else:
        return "hear_right" if random.random() < LISTEN_ACCURACY else "hear_left"


# 奖励函数
def get_reward(state, action):
    # 给定真实状态和动作，返回奖励
    if action == "listen":
        return LISTEN_COST
    if action == "open_left":
        if state == "tiger_left":
            return OPEN_TIGER_PENALTY
        else:
            return OPEN_TREASURE_REWARD
    if action == "open_right":
        if state == "tiger_right":
            return OPEN_TIGER_PENALTY
        else:
            return OPEN_TREASURE_REWARD
    raise ValueError(f"Unknown action: {action}")


# 定义belief
belief = {"tiger_left": 0.5, "tiger_right": 0.5}


# belief update
# 给定状态下听到某个观测的概率
def observation_prob(observation, state, action):
    # P（o| s, a）
    if action != "listen":
        return 1.0 if observation is None else 0.0
    if state == "tiger_left":
        if observation == "hear_left":
            return LISTEN_ACCURACY
        elif observation == "hear_right":
            return 1.0 - LISTEN_ACCURACY
    if state == "tiger_right":
        if observation == "hear_right":
            return LISTEN_ACCURACY
        elif observation == "hear_left":
            return 1.0 - LISTEN_ACCURACY
    return 0.0


# Bayes更新
# 当前belief + 动作 + 新观测， 计算新的belief分布
# 输入：belief 当前对状态的概率分布 action: 刚执行的动作 observation: 刚获得的观测
def update_belief(belief, action, observation):
    # 根据动作和观测更新belief
    if action != "listen":
        return belief.copy()
    new_belief = {}
    # listen 不改变状态
    for state in STATES:
        new_belief[state] = observation_prob(observation, state, action) * belief[state]
    # 归一化
    total = sum(new_belief.values())
    for state in STATES:
        new_belief[state] /= total
    return new_belief


def demo_one_listen():
    # 随机生成真实状态
    true_state = random.choice(STATES)

    # 初始 belief
    belief = {"tiger_left": 0.5, "tiger_right": 0.5}

    print("真实状态:", true_state)
    print("初始 belief:", belief)

    action = "listen"
    observation = sample_observation(true_state, action)
    reward = get_reward(true_state, action)

    print("动作:", action)
    print("观测:", observation)
    print("奖励:", reward)

    belief = update_belief(belief, action, observation)
    print("更新后 belief:", belief)


def simple_policy(belief, threshold=0.8):
    p_left = belief["tiger_left"]
    p_right = belief["tiger_right"]

    if p_left >= threshold:
        return "open_right"  # 老虎大概率在左，开右门
    elif p_right >= threshold:
        return "open_left"  # 老虎大概率在右，开左门
    else:
        return "listen"


def run_episode(verbose=True):
    true_state = random.choice(STATES)
    belief = {"tiger_left": 0.5, "tiger_right": 0.5}

    total_reward = 0
    step = 0

    if verbose:
        print("=" * 50)
        print("新回合开始")
        print("真实状态:", true_state)
        print("初始 belief:", belief)

    while True:
        step += 1
        action = simple_policy(belief, threshold=0.8)
        reward = get_reward(true_state, action)
        total_reward += reward

        if verbose:
            print(f"\nStep {step}")
            print("当前 belief:", belief)
            print("选择动作:", action)
            print("即时奖励:", reward)

        if action == "listen":
            observation = sample_observation(true_state, action)
            if verbose:
                print("观测结果:", observation)

            belief = update_belief(belief, action, observation)

        else:
            if verbose:
                if reward == OPEN_TREASURE_REWARD:
                    print("结果: 打开了宝藏门")
                else:
                    print("结果: 打开了老虎门")
                print("回合结束，总奖励:", total_reward)
            break

    return total_reward


run_episode()
