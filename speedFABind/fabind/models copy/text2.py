import torch
import time
from torch.cuda.amp import autocast, GradScaler
import torch.profiler.profiler as profiler

# 确保GPU可用
if not torch.cuda.is_available():
    raise ValueError("GPU is required for this script.")

# 定义矩阵尺寸
M = 30380
N = 1024
K = 512

# 生成随机矩阵
A = torch.randn(M, K, device='cuda', requires_grad=True)
B = torch.randn(K, N, device='cuda', requires_grad=True)
C = torch.zeros(M, N, device='cuda', requires_grad=True)
with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
                profile_memory=True,
                # with_stack=True,
                record_shapes=True
            ) as prof:
# 1. 使用torch.addmm进行常规单精度矩阵乘法（作为基准对比）
                def torch_addmm_benchmark():
                    start_time = time.time()
                    result_addmm = torch.addmm(C, A, B)
                    end_time = time.time()
                    duration = end_time - start_time
                    return result_addmm, duration

                # 2. 使用Tensor Core通过自动混合精度进行矩阵乘法
                def tensor_core_benchmark():
                    scaler = torch.amp.GradScaler('cuda')
                    start_time = time.time()
                    with torch.amp.autocast('cuda'):
                        result_tensor_core = torch.addmm(C, A, B)
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    return result_tensor_core, duration

                # 3. 使用PyTorch中利用CUTLASS优化的矩阵乘法（在合适条件下PyTorch会自动调用CUTLASS加速）
                def cutlass_benchmark():
                    start_time = time.time()
                    result_cutlass = torch.matmul(A, B)  # PyTorch的matmul在合适场景下会利用CUTLASS等优化
                    end_time = time.time()
                    duration = end_time - start_time
                    return result_cutlass, duration

                # 执行对比并输出结果
                if __name__ == "__main__":
                    # torch.addmm 计算
                    result_addmm, time_addmm = torch_addmm_benchmark()
                    print(f"torch.addmm计算耗时: {time_addmm} 秒")

                    # Tensor Core 计算
                    result_tensor_core, time_tensor_core = tensor_core_benchmark()
                    print(f"Tensor Core计算耗时: {time_tensor_core} 秒")

                    # CUTLASS 计算
                    result_cutlass, time_cutlass = cutlass_benchmark()
                    print(f"CUTLASS计算耗时: {time_cutlass} 秒")

                    # 比较精度（计算结果的最大绝对差值）
                    max_diff_addmm_tensor_core = torch.max(torch.abs(result_addmm - result_tensor_core)).item()
                    max_diff_addmm_cutlass = torch.max(torch.abs(result_addmm - result_cutlass)).item()
                    print(f"torch.addmm与Tensor Core结果最大绝对差值: {max_diff_addmm_tensor_core}")
                    print(f"torch.addmm与CUTLASS结果最大绝对差值: {max_diff_addmm_cutlass}")