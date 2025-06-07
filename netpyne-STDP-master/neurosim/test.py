# test_aigame.py
from aigame2 import AIGame as AIGameGym
from aigame import AIGame as AIGameBrax
from game_interface import GameInterface
import numpy as np
import random

# 配置字典（包含observation_map和net配置）
gym_config = {
    'env': {
        'name': 'CartPole-v1',
        'render': False,
        'seed': 42,
        'observation_map': [
            {'type': 'linear'},  # cart_position
            {'type': 'sigmoid'},  # pole_angle
            {'type': 'linear'},  # cart_velocity
            {'type': 'sigmoid_div', 'scale': 2.0}  # pole_angular_velocity
        ]
    },
    'actionsPerPlay': 1,
    'observationsToKeep': 100,
    'rewardsToKeep': 100,
    'net': {
        'InputMaxRate': 100.0,
        'inputPop': 'Input',
        'allpops': {'Input': 16}  # 输入神经元数量
    }
}

brax_config = {
    'env': {
        'name': 'inverted_pendulum',
        'render': False,
        'seed': 42,
        'observation_map': [
            {'type': 'linear'},  # cart_position
            {'type': 'sigmoid'},  # pole_angle
            {'type': 'linear'},  # cart_velocity
            {'type': 'sigmoid_div', 'scale': 2.0}  # pole_angular_velocity
        ]
    },
    'actionsPerPlay': 1,
    'observationsToKeep': 100,
    'rewardsToKeep': 100,
    'net': {
        'InputMaxRate': 100.0,
        'inputPop': 'Input',
        'allpops': {'Input': 16}
    }
}


def test_aigame(module_name, game_class, config):
    print(f"\n=== Testing {module_name} ===")

    # 初始化AIGame
    game = game_class(config)

    # 测试初始观察值
    print(f"Initial observation: {game.observations[-1]}")
    print(f"Observation shape: {np.array(game.observations[-1]).shape}")
    print(f"Observation type: {type(game.observations[-1])}")

    # 测试observation_space
    obs_space = game.env.observation_space
    print(f"Observation space low: {obs_space.low}")
    print(f"Observation space high: {obs_space.high}")

    # 测试随机动作
    random_action = game.randmove()
    print(f"Random action: {random_action}")
    print(f"Action type: {type(random_action)}")

    # 执行游戏步骤（随机两步）
    # actions = [game.randmove(), game.randmove()]
    # print(f"Actions to play (random): {actions}")
    # rewards, done = game.playGame(actions)
    # print(f"Rewards: {rewards}")
    # print(f"Done: {done}")

    # 测试GameInterface
    game_interface = GameInterface(game, config)
    firing_rates = game_interface.input_firing_rates()
    print(f"Firing rates: {firing_rates}")
    print(f"Firing rates shape: {firing_rates.shape}")
    print(f"Firing rates type: {type(firing_rates)}")

    # # 检查状态更新
    # print(f"Updated observations (last): {game.observations[-1]}")
    # print(f"Updated rewards (last): {game.rewards[-1]}")
    # print(f"Step count: {game.count_steps}")
    # print(f"Episode count: {game.count_episodes}")


def test_fixed_left_steps(module_name, game_class, config):
    print(f"\n=== Testing {module_name} with Fixed Left Actions Until Done ===")

    # 初始化AIGame
    game = game_class(config)
    step = 0

    while True:
        # 固定向左动作（Gym: 0, Brax: 映射为0）
        actions = [random.randint(0, 1)]
        rewards, done = game.playGame(actions)

        step += 1  # 每次执行2步
        print(f"Step {step}:")
        print(f"  Observation: {game.observations[-1]}")
        print(f"  Rewards: {rewards}")
        print(f"  Done: {done}")

        if done:
            print(f"Game ended after {step} steps")
            print(f"Final step count: {game.count_steps}")
            print(f"Episode count: {game.count_episodes}")
            break


if __name__ == "__main__":

    # 测试Gym版本
    test_aigame("aigame2.py (Gym)", AIGameGym, gym_config)
    test_fixed_left_steps("aigame2.py (Gym)", AIGameGym, gym_config)

    # 测试Brax版本
    test_aigame("aigame.py (Brax)", AIGameBrax, brax_config)
    test_fixed_left_steps("aigame.py (Brax)", AIGameBrax, brax_config)