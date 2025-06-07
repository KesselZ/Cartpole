import os
import random
from omegaconf import OmegaConf
from tqdm import tqdm

# 你的老代码模块
import es_dense_acrobot


def run_experiments(gpu_id: int, num_runs: int, base_seed: int = None):
    """
    顺序运行 es_dense_cartpole.py 的 main 函数，指定 GPU，每次运行使用不同的 seed。
    确保上一个实验结束后再开始下一个。

    Args:
        gpu_id (int): 使用的 GPU ID
        num_runs (int): 运行次数
        base_seed (int, optional): 基础种子，默认随机生成
    """
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running experiments sequentially on GPU {gpu_id}")

    # 生成基础种子
    if base_seed is None:
        base_seed = 1
    print(f"Base seed: {base_seed}")

    # 顺序运行实验
    for i in tqdm(range(num_runs), desc="Running experiments"):
        seed = base_seed + i  # 每次运行递增 seed
        conf = OmegaConf.create({"seed": seed})  # 只传入 seed，其他参数由老代码默认处理

        print(f"\nStarting run {i + 1}/{num_runs} with seed={seed}")
        # 调用老代码的 main 函数，等待其完成
        es_dense_acrobot.main(conf)
        print(f"Finished run {i + 1}/{num_runs} with seed={seed}")


def main():
    # 配置参数
    gpu_id = 1  # 指定 GPU ID
    num_runs = 20  # 运行次数

    # 运行实验
    run_experiments(gpu_id=gpu_id, num_runs=num_runs)


if __name__ == "__main__":
    main()