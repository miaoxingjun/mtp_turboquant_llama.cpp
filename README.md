<img width="1920" height="1080" alt="c2000ba7bc722d28438e7569d70b8875" src="https://github.com/user-attachments/assets/1f19bb13-9697-4f81-8c71-2d7d2a62e44f" />
<img width="1920" height="1080" alt="de79ca7e4d70339f174a50309d5ba93e" src="https://github.com/user-attachments/assets/2e3cfb7a-bdb2-45e0-b957-7d0491891122" />
<img width="1920" height="1080" alt="2911fa6c5c9a83d7b6fc23d1170943df" src="https://github.com/user-attachments/assets/f6d8c9bf-dbe4-4c77-882c-6d3bc0dc6577" />


# llama.cpp Turbo+MTP 增强版

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的合并版本，集成了 **TurboQuant** 加速支持和 **MTP（Multi-Token Prediction）** 功能。

## 特性

- **TurboQuant KV Cache 量化加速**
  - `TURBO3` — 3-bit TurboQuant + WHT
  - `TURBO4` — 4-bit TurboQuant + WHT
- **MTP（Multi-Token Prediction）多 Token 预测**
  - 支持 Qwen3.5 / Qwen3.6 系列模型的 speculative decoding
  - 通过 `--spec-type draft-mtp` 启用
- **CUDA 后端完整支持**
  - 所有 TurboQuant 量化类型均有 CUDA kernel 加速
  - Flash Attention 支持 TURBO 量化 KV Cache
- 保留 llama.cpp 全部原有功能（GGUF 格式、多后端、API 服务器等）

## 编译指南（Windows + CUDA）

### 前置要求

- CMake 3.21+
- Visual Studio 2022（或 Ninja）
- NVIDIA CUDA Toolkit 12.x
- 支持 CUDA 的 NVIDIA GPU

### 编译命令

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j 8
```

### 完整编译（所有工具）

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8
```

## 使用示例

### 启动 API 服务器（启用 TurboQuant）

```bash
build\bin\Release\llama-server.exe ^
  -m model-Q4_K_M.gguf ^
  --cache-type-k turbo4 ^
  --cache-type-v turbo3 ^
  -c 8192
```

### 启用 MTP Speculative Decoding

```bash
build\bin\Release\llama-server.exe ^
  -m model-Q4_K_M.gguf ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 2
  -c 8192
```

### CLI 对话模式

```bash
build\bin\Release\llama-cli.exe ^
  -m model-Q4_K_M.gguf ^
  --cache-type-k turbo4 ^
  --cache-type-v turbo4 ^
  -n 512 -p "你好，请介绍一下你自己"
```

## 支持的模型

### MTP 支持

| 模型系列 | 说明 |
|---------|------|
| Qwen3.5 / Qwen3.6 | 原生 MTP 头支持 |
| Qwen3.5-MoE | MoE 架构 MTP 支持 |

### 通用模型

所有 llama.cpp 支持的模型均可使用 TurboQuant KV Cache 量化功能，包括但不限于：

- Llama / Mistral / Qwen 系列
- DeepSeek / Phi / Gemma 系列
- MoE 架构模型（Mixtral、Qwen-MoE 等）

## 注意事项

- TurboQuant 量化类型（`TURBO2/3/4`）主要用于 **KV Cache**，通过 `--cache-type-k` 和 `--cache-type-v` 参数设置
- MTP 功能需要配套的 MTP head 模型文件，可通过 `-hf` 参数自动下载
- CUDA 后端对 TurboQuant 有完整 kernel 支持，CPU 后端同样可用但性能较低
- 使用 TURBO 量化可以显著减少 KV Cache 显存占用并提升 decode 吞吐量，但可能有轻微精度损失

## 鸣谢

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 本项目的基础框架
- [ggml](https://github.com/ggml-org/ggml) — 底层张量计算库
- TurboQuant 和 MTP 的原始贡献者们
