import torch
import time
import torch.amp
import torch.profiler.profiler as profiler

from FlagGems.src.flag_gems.ops.addmm import addmm

# 生成两个随机张量，形状为(3, 3)
A = torch.randn(30380, 512).cuda()
B = torch.randn(512, 1024).cuda()
C = torch.randn(1024).cuda()
# 定义addmm参数
alpha = 1.0
beta = 1.0

# # 原始PyTorch计算
# start_time_pytorch = time.time()
# result_pytorch = torch.addmm(beta * C, alpha * A, B)
# end_time_pytorch = time.time()
# # 保存原始结果用于精度比较
# reference_result = result_pytorch.clone().detach()
# # 计算原始PyTorch耗时
# pytorch_duration = end_time_pytorch - start_time_pytorch
# print("原始PyTorch计算耗时：", pytorch_duration)


# from torch.cuda.amp import autocast

# # 启用Tensor Core优化计算
# start_time_tensor_core = time.time()
# with autocast():
#     result_tensor_core = torch.addmm(beta * C, alpha * A, B)
# end_time_tensor_core = time.time()
# # 计算Tensor Core优化耗时
# tensor_core_duration = end_time_tensor_core - start_time_tensor_core
# print("Tensor Core优化计算耗时：", tensor_core_duration)
# # 比较精度
# difference = torch.max(torch.abs(reference_result - result_tensor_core))
# print("最大精度差值：", difference)

# precision_threshold = 1e - 6
# if difference < precision_threshold:
#     print("精度损失在可接受范围内")
# else:
#     print("可能存在较大精度损失")

# if tensor_core_duration < pytorch_duration:
#     print("Tensor Core优化在速度上有提升")
# elif tensor_core_duration > pytorch_duration:
#     print("Tensor Core优化在速度上没有提升")
# else:
#     print("Tensor Core优化和原始PyTorch速度相同")




num_runs = 10
pytorch_durations = []
tensor_core_durations = []
triton_durations = []
with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
                profile_memory=True,
                # with_stack=True,
                record_shapes=True
            ) as prof:
    for _ in range(num_runs):

        with torch.profiler.record_function("torch"): 
            # 原始PyTorch计算
            start_time_pytorch = time.time()
            result_pytorch = torch.addmm(C, A, B)
            end_time_pytorch = time.time()
            pytorch_durations.append(end_time_pytorch - start_time_pytorch)

        with torch.profiler.record_function("tensor_core"): 
            # 启用Tensor Core优化计算
            start_time_tensor_core = time.time()
            with torch.amp.autocast(device_type='cuda'):
                result_tensor_core = torch.addmm(C, A, B)
            end_time_tensor_core = time.time()
            tensor_core_durations.append(end_time_tensor_core - start_time_tensor_core)

        with torch.profiler.record_function("triton"): 
            start_time_pytorch = time.time()
            result_triton = addmm(C, A, B)
            end_time_pytorch = time.time()
            triton_durations.append(end_time_pytorch - start_time_pytorch)
        difference = torch.max(torch.abs(result_pytorch - result_tensor_core))
        print("tensor最大精度差值：", difference)
        difference = torch.max(torch.abs(result_pytorch - result_triton))
        print("triton最大精度差值：", difference)

average_pytorch_duration = sum(pytorch_durations) / num_runs
average_tensor_core_duration = sum(tensor_core_durations) / num_runs
average_triton_duration = sum(triton_durations) / num_runs
print("原始PyTorch平均计算耗时：", average_pytorch_duration)
print("Tensor Core优化平均计算耗时：", average_tensor_core_duration)
print("triton优化平均计算耗时：", average_triton_duration)


