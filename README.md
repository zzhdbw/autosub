# JA2CN — 日本语视频音频 → 中文字幕

使用 FunASR（SenseVoiceSmall）对日语视频进行语音识别，通过 Silero VAD 切分语音片段，再经 HY-MT1.5-1.8B（llama.cpp GGUF）翻译成中文，生成标准 `.srt` 字幕文件。

## 工作流程

```
视频文件 → PyAV 提取音频 → Silero VAD 语音分段
→ FunASR SenseVoiceSmall 逐段识别 → HY-MT1.5-1.8B GGUF 日译中 → SRT 字幕文件
```

VAD（语音活动检测）确保每个语音片段独立识别，不会合并成长句字幕。

## 环境要求

- **Python** >= 3.10

## 安装

```bash
# 克隆项目后进入目录
cd JA2CN

# 创建虚拟环境（推荐）
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖（llama.cpp GGUF 为默认翻译后端）
uv --cache-dir ~/.uv/cache pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

## 模型

模型会自动下载到 `model/` 目录：

| 模型 | 大小 | 用途 |
|------|------|------|
| Silero VAD | ~2 MB | 语音活动检测（语言无关） |
| SenseVoiceSmall | ~500 MB | 日语音声识别 |
| HY-MT1.5-1.8B GGUF | ~1.1 GB | 日本语→中文翻译（默认，llama.cpp） |
| HY-MT1.5-1.8B（Transformers） | ~3.8 GB | 日本语→中文翻译（`--backend transformers`） |

GGUF 模型已内置在仓库 `model/` 目录下，开箱即用。如需 Transformers 后端，模型会自动下载。

## 使用

```bash
cd JA2CN
source .venv/bin/activate

# 完整流程（默认 llama.cpp GGUF 后端）
ja2cn data/视频文件名.mp4

# 指定输出路径
ja2cn data/视频文件名.mp4 -o output/字幕.srt

# 仅 VAD + ASR（不翻译，生成日语字幕）
ja2cn data/视频文件名.mp4 --skip-translate

# 跳过 VAD（复用已有 VAD 分段结果）
ja2cn data/视频文件名.mp4 --skip-vad

# 跳过 VAD 和 ASR（仅翻译已有 segments）
ja2cn data/视频文件名.mp4 --skip-vad --skip-asr

# 调试模式（显示 DEBUG 日志）
ja2cn data/视频文件名.mp4 --verbose

# 如需 Transformers 后端
ja2cn data/视频文件名.mp4 --backend transformers
```

每个步骤均可独立运行，支持断点续跑。

## 输出结构

```
output/
├── video_name.wav                   # 提取的音频
├── video_name_vad.json              # VAD 分段结果（时间戳）
├── video_name_segments.json          # ASR 识别结果（日语原文 + 时间戳）
├── video_name_translated.json        # 翻译结果
└── video_name.srt                    # 最终字幕文件
```

中间结果保存为 JSON，配合 `--skip-*` 参数可跳过已完成的步骤。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | (必填) | 输入视频文件 |
| `-o, --output` | `output/<输入名>.srt` | 输出 SRT 路径 |
| `--output-dir` | `output` | 中间文件输出目录 |
| `--model-dir` | `model` | 模型存储根目录 |
| `--prompt-template` | `将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}` | 翻译模型的 prompt 模板 |
| `--temperature` | `0.7` | 采样温度 |
| `--top-k` | `20` | Top-k 采样 |
| `--top-p` | `0.6` | Top-p 采样 |
| `--repetition-penalty` | `1.05` | 重复惩罚系数 |
| `--backend` | `llamacpp` | 翻译后端：`llamacpp`（默认，GGUF 1.1 GB）或 `transformers`（3.8 GB） |
| `--gguf-model` | `model/Tencent-Hunyuan/HY-MT1.5-1.8B-GGUF/HY-MT1.5-1.8B-Q4_K_M.gguf` | GGUF 模型路径 |
| `--chunk-ms` | `10000` | ASR 分段时长（毫秒，仅无 VAD 时使用） |
| `-v, --verbose` | — | 开启 DEBUG 级别日志 |
| `--skip-vad` | — | 跳过 VAD（复用已有 vad.json） |
| `--skip-asr` | — | 跳过 ASR（复用已有 segments.json） |
| `--skip-translate` | — | 跳过翻译（仅输出日语字幕） |

## 注意事项

1. **首次运行需要下载模型**（SenseVoiceSmall ~500 MB），请保持网络畅通
2. **llama.cpp GGUF 后端**为默认，模型已内置，开箱即用，内存要求 ~2 GB
3. **Transformers 后端**（`--backend transformers`）模型约 3.8 GB，需要额外下载，CPU 推理较慢，内存建议 ≥ 16 GB
