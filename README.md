# JA2CN — 日本语视频音频 → 中文字幕

使用 FunASR（SenseVoiceSmall）对日语视频进行语音识别，通过 Silero VAD 切分语音片段，再经 HY-MT1.5-1.8B 模型翻译成中文，生成标准 `.srt` 字幕文件。

## 工作流程

```
视频文件 → ffmpeg 提取音频 → Silero VAD 语音分段
→ FunASR SenseVoiceSmall 逐段识别 → HY-MT1.5-1.8B 日译中 → SRT 字幕文件
```

VAD（语音活动检测）确保每个语音片段独立识别，不会合并成长句字幕。

## 环境要求

- **Python** >= 3.10
- **ffmpeg**（音频提取）

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

## 安装

```bash
# 克隆项目后进入目录
cd JA2CN

# 创建虚拟环境（推荐）
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装基础依赖（Transformers 后端，默认）
uv --cache-dir ~/.uv/cache pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

# 如需 llama.cpp 后端（更快，需额外安装）
uv --cache-dir ~/.uv/cache pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[llamacpp]"
```

## 模型下载

模型会自动下载到 `model/` 目录：

| 模型 | 大小 | 用途 |
|------|------|------|
| Silero VAD | ~2 MB | 语音活动检测（语言无关） |
| SenseVoiceSmall | ~500 MB | 日语音声识别 |
| HY-MT1.5-1.8B | ~3.8 GB | 日本语→中文翻译 |

首次运行时会自动下载。翻译模型较大（~3.8 GB），也可提前下载：

```bash
source .venv/bin/activate
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Tencent-Hunyuan/HY-MT1.5-1.8B', cache_dir='model')
"
```

## 使用

```bash
cd JA2CN
source .venv/bin/activate

# 完整流程
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

# 使用 llama.cpp 后端（GGUF，默认 — 更快，无需 HuggingFace）
ja2cn data/视频文件名.mp4

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
| `--backend` | `llamacpp` | 翻译后端：`llamacpp`（默认）或 `transformers` |
| `--gguf-model` | `model/Tencent-Hunyuan/HY-MT1.5-1.8B-GGUF/HY-MT1.5-1.8B-Q4_K_M.gguf` | GGUF 模型路径 |
| `--chunk-ms` | `10000` | ASR 分段时长（毫秒，仅无 VAD 时使用） |
| `-v, --verbose` | — | 开启 DEBUG 级别日志 |
| `--skip-vad` | — | 跳过 VAD（复用已有 vad.json） |
| `--skip-asr` | — | 跳过 ASR（复用已有 segments.json） |
| `--skip-translate` | — | 跳过翻译（仅输出日语字幕） |

## 注意事项

1. **首次运行需要下载模型**，请保持网络畅通
2. **CPU 运行速度较慢**（HY-MT1.5-1.8B 为 1.8B 参数），建议预留充足时间
3. **内存要求**：建议 ≥ 16 GB RAM
4. **HY-MT1.5-1.8B 的 prompt 格式** 可能需要根据模型实际要求调整，通过 `--prompt-template` 参数自定义
