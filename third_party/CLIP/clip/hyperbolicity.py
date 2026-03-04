import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from itertools import combinations


def euclidean_hyperbolicity(points):
    """
    计算点集的欧式双曲性（原始逻辑不变，新增输入适配）
    Args:
        points: numpy数组 / torch张量（CPU/CUDA），形状 [n, d]
    Returns:
        max_hyp: 最大双曲性值
    """
    # 新增：适配torch张量（转CPU+detach+numpy）
    points = _tensor_to_numpy(points)

    # 边界检查：点数不足4个时返回0（无法计算4点组合）
    if points.shape[0] < 4:
        return 0.0

    # 原有逻辑：计算成对欧式距离
    distances = squareform(pdist(points, 'euclidean'))

    n = points.shape[0]
    max_hyp = 0

    # 遍历所有4点组合计算双曲性
    for combo in combinations(range(n), 4):
        d = [distances[combo[i], combo[j]] for i in range(4) for j in range(i + 1, 4)]
        d.sort()
        hyp = (d[4] + d[5] - d[2] - d[3]) / 2
        max_hyp = max(max_hyp, hyp)

    return max_hyp


def hyperbolicity_sample_euclidean(points, num_samples=5000):
    """
    采样计算归一化双曲性（修复张量适配+边界检查+性能优化）
    Args:
        points: numpy数组 / torch张量（CPU/CUDA），形状 [n, d]
        num_samples: 采样次数（默认5000，避免原50000导致计算过慢）
    Returns:
        normalized_hyp: 归一化后的双曲性
    """
    # 新增：适配torch张量
    points = _tensor_to_numpy(points)

    # 边界检查：点数不足4个时返回0
    if points.shape[0] < 4:
        return 0.0

    # 原有逻辑：计算成对距离和直径
    distances = squareform(pdist(points, 'euclidean'))
    diam = np.max(distances)

    # 边界检查：直径为0（所有点重合）时返回0
    if diam == 0:
        return 0.0

    n = points.shape[0]
    max_hyp = 0

    # 优化：限制采样次数不超过所有可能的4点组合数
    max_possible = np.math.comb(n, 4) if n >= 4 else 0
    num_samples = min(num_samples, max_possible)

    # 原有逻辑：随机采样4点组合计算双曲性
    for _ in range(num_samples):
        combo = np.random.choice(n, 4, replace=False)
        d = [distances[combo[i], combo[j]] for i in range(4) for j in range(i + 1, 4)]
        d.sort()
        hyp = (d[4] + d[5] - d[2] - d[3]) / 2
        max_hyp = max(max_hyp, hyp)

    # 归一化双曲性
    return 2 * max_hyp / diam


def multiple_trials_hyperbolicity(points, num_samples=5000, num_trials=10):
    """
    多次采样计算平均双曲性（原始逻辑不变）
    """
    hyperbolicities = []

    for _ in range(num_trials):
        hyp = hyperbolicity_sample_euclidean(points, num_samples)
        hyperbolicities.append(hyp)

    mean_hyperbolicity, std_hyperbolicity = np.mean(hyperbolicities), np.std(hyperbolicities)
    return mean_hyperbolicity, std_hyperbolicity


def mean_hyperbolicity_per_batch(points_tensor, num_samples=5000, num_trials=10):
    """
    批量计算每个样本的平均双曲性（核心修改：适配torch张量输入）
    Args:
        points_tensor: numpy数组 / torch张量（CPU/CUDA），形状 [batch, n, d]
        num_samples: 单次采样数（默认从50000下调到5000，避免计算过慢）
        num_trials: 试验次数
    Returns:
        batch_hyperbolicities_means: 每个batch的平均双曲性列表
    """
    np.random.seed(42)
    batch_hyperbolicities_means = []

    # 新增：适配torch张量输入（处理CUDA/CPU）
    points_tensor = _tensor_to_numpy(points_tensor)

    # 原有逻辑：遍历每个batch计算双曲性
    for batch_idx in range(points_tensor.shape[0]):
        points = points_tensor[batch_idx]
        mean_hyperbolicity, _ = multiple_trials_hyperbolicity(points, num_samples, num_trials)
        batch_hyperbolicities_means.append(mean_hyperbolicity)

    return batch_hyperbolicities_means


# ---------------------- 新增核心工具函数 ----------------------
def _tensor_to_numpy(x):
    """
    统一将输入转为numpy数组（处理torch张量/CUDA张量）
    Args:
        x: numpy数组 / torch张量（CPU/CUDA）
    Returns:
        x_np: numpy数组（CPU）
    """
    if isinstance(x, torch.Tensor):
        # 1. 从计算图分离 + 移到CPU + 转为numpy（解决CUDA张量报错核心）
        x_np = x.detach().cpu().numpy()
    else:
        # 2. 非张量直接转numpy
        x_np = np.asarray(x)
    return x_np


if __name__ == "__main__":
    """
    测试示例：兼容numpy/torch/CUDA张量输入，验证双曲性计算
    双曲性数值越小，空间越接近完美树状结构
    """
    # 1. 生成测试数据（模拟ViT输出：[batch=4, n=256, d=4096]）
    n, d = 256, 4096
    # 测试1：numpy数组输入（原有逻辑）
    points_np = np.random.rand(4, n, d)

    # 测试2：torch CPU张量输入
    points_torch_cpu = torch.tensor(points_np)

    # 测试3：torch CUDA张量输入（模拟训练时的真实场景）
    points_torch_cuda = points_torch_cpu.cuda() if torch.cuda.is_available() else points_torch_cpu

    # 2. 计算双曲性（兼容所有输入类型）
    import time

    for name, points in [
        ("numpy数组", points_np),
        ("torch CPU张量", points_torch_cpu),
        ("torch CUDA张量", points_torch_cuda)
    ]:
        start_time = time.time()
        mean_hyp = mean_hyperbolicity_per_batch(points, num_samples=5000, num_trials=10)
        print(f"\n{name}计算结果：")
        print(f"各batch平均双曲性：{mean_hyp}")
        print(f"耗时：{time.time() - start_time:.2f}秒")