# LichtFeld-Studio CoreX (ivcore11) 迁移会话记录

## 来源
- 上游仓库：https://github.com/MrNeRF/LichtFeld-Studio.git
- 迁移起点 commit：`7efd48d58771c26456465bf35da7200754b739f5`（分支 `master`，提交日期 2026-07-23）
- 项目性质：3D Gaussian Splatting（3DGS）辐射场训练/可视化工作站，C++23 + CUDA，CMake + vcpkg 构建。
  上游默认目标平台是 NVIDIA GPU + CUDA 12.8 + Vulkan，桌面 GUI 应用。

## CUDA 使用性质
- **关键特征：本项目是 LibTorch-free 的**，自带一套定制 CUDA 张量库（`src/core/tensor/*.cu`，
  目标库 `lfs_tensor_kernels`），并不依赖 libtorch，因此核心计算路径不会拖入庞大的 libtorch CUDA 构建。
- 全仓共 67 个 `.cu`，涵盖：核心张量核（`src/core/tensor`）、核心 CUDA 工具（`src/core/cuda`）、
  训练核与光栅化（`src/training`，含 gsplat / fastgs）、IO、渲染（含 CUDA-Vulkan interop）等。
- CUDA 编译通过 CMake `project(... CUDA)` 走 nvcc；CoreX 上必须改用 `clang++ -x ivcore`
  （`--cuda-gpu-arch=ivcore11`），红线禁止使用 nvcc / 触碰 `/usr/local/corex`。

## 环境
- GPU：2× Iluvatar BI-V150（每卡 32 GB），驱动 4.5.0，`ixsmi` 记录于 `env/ixsmi.log`。
  分配 GPU 0（`CUDA_VISIBLE_DEVICES=0`），全程未超过 2 卡。
- 工具链：CoreX 4.5.0，clang/clang++ 22.1.0git（`env/clang_version.txt`）。
- 注意：在受限沙箱内 `ixsmi` 报 "No supported GPUs found"（`/dev/iluvatar*` 被沙箱屏蔽）；
  以完整权限运行后正常识别两张 BI-V150。所有编译/跑 GPU 命令均在可访问设备的环境下执行。

## 适配内容（Failure Gate：均为实测复现后修复，非推断）
本次聚焦"CUDA 核 + 核心库"里最内聚、可独立验证的 **LibTorch-free 张量核库 `lfs_tensor_kernels`（10 个 `.cu`）**，
用 `clang++ -x ivcore` 逐个复现编译墙并修复：

1. **NV PTX 内联汇编 `l` 约束（packed128 流式访存）**：`load128cs/store128cs/store128cg`
   调用 `__ldcs/__stcs/__stcg`，在 ivcore11 后端 llc 阶段报 `unknown asm constraint 'l'`（`-O0/-O3` 均崩）。
   `__ILUVATAR__` 分支回退为普通 128-bit `int4` load/store（仅去掉缓存提示，语义等价）。
2. **ixthrust 无 `par_nosync`**（版本滞后，非平台 bug）：`tensor_ops.cu` 用 `__ILUVATAR__` 守卫
   将 `thrust::cuda::par_nosync` 别名到同步的 `par`（只是恢复每次调用的隐式同步）。
3. **`__device__`-only 仿函数的 `invoke_result` 推导失败**：`broadcast_index_functor::operator()`
   仅 `__device__`，clang-cuda 在 host pass 用 `std::invoke_result` 推导 `transform_iterator` 值类型时失败；
   改为 `__host__ __device__`（函数体纯整数运算，host 安全）。
4. **`__half` 与整型 static_cast 二义**：ivcore11 上 `__half` 只有 float/double 转换，`ConvertFunctor`
   在 half<->整型间用 `if constexpr` 经 `float` 中转消歧。
5. **CoreX cub `DeviceSegmentedReduce::Reduce` 签名差异**：CoreX 版对 begin/end 用同一个 `OffsetIteratorT`，
   而源码传了两个不同 lambda 的 `transform_iterator` 类型导致模板推导失败；改用单一 offset 迭代器
   `offsets / offsets+1`（cub 标准写法，NV 亦兼容）。
6. **nvcc → clang 编译选项翻译**：`-use_fast_math` → `-ffast-math`；`--extended-lambda` /
   `--expt-relaxed-constexpr` 直接去掉（clang-cuda 原生支持）；arch `sm_XX` → `ivcore11`。

## 结果
- **编译**：`lfs_tensor_kernels` 全部 **10/10** `.cu` 在 ivcore11 编译成功（`build/compile.log`、`build/obj/*.o`）。
- **测试（on-GPU 冒烟）**：自包含冒烟 `test/smoke.cu` 直接 `#include` 迁移后的设备头
  （packed128.cuh / warp_reduce.cuh / tensor_functors.hpp）并在 GPU 0 上启动核，验证数值，
  **6/6 全部通过**（含 `load128cs` 的 `__ldcs` 修复路径、warpSize=64 下的 warp/block 归约、functor 归约、
  广播索引映射）。仅链接 cudart。设备识别为 `Iluvatar BI-V150 warpSize=64`。
- warpSize=64：源码 warp 归约按 32 假设（offset 从 16 起、`lane=tid%32`），在测试规模（32/64/256 线程）
  下数值正确；作为潜在 warp32→warp64 语义风险点记录，供后续更大 block/跨 warp 混合场景复核。

## Failure Gate / 范围说明（诚实记录）
- **已验证范围**：LibTorch-free CUDA 张量核库（编译 + on-GPU 冒烟）。
- **超出本节点范围（blocker #6，非 terminal，属环境/工程量约束）**：
  完整应用与仓库自带 gtest/ctest 正式测试集（200+ 用例）需要完整 vcpkg 依赖图
  （imgui/sdl3/vulkan/usd/ffmpeg/openimageio/python/glm/spdlog/tbb/gtest）以及 host 端 `lfs_core`
  的内存池/错误上报/VramProfiler/Tensor 基础设施——把张量核对象做 host 链接会把整个 `lfs_core` + vcpkg
  依赖拉进来；且 GUI/可视化依赖 NVIDIA + Vulkan interop，无法在无显示的 CoreX 节点 headless 运行。
  这些属"需要另建完整 vcpkg 工具链"的工程量，不是 ivcore11 的硬件/编译器 terminal 墙；已按红线保留、
  未 cherry-pick、未虚报，正式测试集计入"超范围"并在此说明。
- 因此 `test_summary.json` 的计数仅覆盖本次确定并完整执行的 CoreX 范围（10 编译 + 6 on-GPU = 16），
  `suite`/`run_command` 已写明范围与命令，无跳过用例（tests_skipped=0）。
