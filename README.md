# AutoSub — Automatic Multi-language Subtitle Generator

语音识别 + 翻译，自动生成多语言字幕。

## 工作流程

```
视频文件 → PyAV 提取音频 → Silero VAD 语音分段
→ FunASR SenseVoiceSmall 逐段识别 → Hy-MT2-1.8B GGUF (STQ 1.25-bit) 翻译 → SRT 字幕文件
```

支持日语 → 中文（可扩展其他语言）。

## 环境要求

- **Python** >= 3.10

## 安装

```bash
cd autosub

# 创建虚拟环境（推荐）
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e .
```

## 模型

首次使用需下载模型，两种方式：

**命令行下载：**
```bash
# Silero VAD（2.2 MB）
curl -L -o model/silero_vad/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/site-packages/silero_vad/data/silero_vad.onnx

# SenseVoiceSmall ONNX（230 MB）
modelscope download --model iic/SenseVoiceSmall-onnx --local_dir model/SenseVoiceSmall-onnx

# Hy-MT2-1.8B GGUF（440 MB）
modelscope download --model Tencent-Hunyuan/Hy-MT2-1.8B-1.25Bit-GGUF \
  Hy-MT2-1.8B-1.25Bit.gguf --local_dir model/Hy-MT2-1.8B-1.25Bit-GGUF
```

**或使用 GUI：** 运行 `autosub-gui`，点击 **Models** → **Download All**。

### 构建 llama.cpp（STQ 量化支持）

Hy-MT2 的 1.25-bit GGUF 需要带 STQ kernel 的 llama.cpp。源码已包含在项目中 `llama.cpp_stq/`：

```bash
# 一键编译
bash scripts/build_llama_stq.sh

# 或手动操作
cd llama.cpp_stq
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(sysctl -n hw.ncpu)
```

| 模型 | 大小 | 用途 |
|------|------|------|
| Silero VAD | ~2 MB | 语音活动检测 |
| SenseVoiceSmall ONNX | ~230 MB | 语音识别 |
| Hy-MT2-1.8B GGUF (STQ 1.25-bit) | ~440 MB | 翻译（默认 llama.cpp） |

## 使用

### 命令行

```bash
# 完整流程
autosub data/视频.mp4

# 仅 VAD + ASR（不翻译）
autosub data/视频.mp4 --skip-translate

# 跳过 VAD / ASR（复用已有结果）
autosub data/视频.mp4 --skip-vad
autosub data/视频.mp4 --skip-vad --skip-asr
```

### 图形界面

```bash
autosub-gui
```

无需命令行操作，支持：
- 文件/目录选择
- 后端切换（llama.cpp / Transformers）
- 分步进度展示
- 模型管理（一键下载）
- 实时日志

## 打包

```bash
pip install pyinstaller
python build.py
open dist/AutoSub.app
```

## 输出结构

```
output/
├── video_name.wav                   # 提取的音频
├── video_name_vad.json              # VAD 分段结果
├── video_name_segments.json          # ASR 识别结果
├── video_name_translated.json        # 翻译结果
└── video_name.srt                    # 最终字幕
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | (必填) | 输入视频文件 |
| `-o, --output` | `output/<输入名>.srt` | 输出 SRT 路径 |
| `--output-dir` | `output` | 中间文件目录 |
| `--backend` | `llamacpp` | 翻译后端 |
| `--skip-vad` | — | 跳过 VAD |
| `--skip-asr` | — | 跳过 ASR |
| `--skip-translate` | — | 跳过翻译 |
| `-v, --verbose` | — | DEBUG 日志 |

## 常见问题

### ONNX 模型加载失败 / Protobuf parsing failed

通常是因为模型文件下载不完整或被覆盖为 HTML 内容（例如下载时遇到 GitHub/ModelScope 重定向页面）。

**Silero VAD：**
```bash
# 检查模型文件是否损坏
file model/silero_vad/silero_vad.onnx
# 应为 "data"，若显示 "HTML document" 则已损坏

# 重新下载
git restore model/silero_vad/silero_vad.onnx
# 或手动重新下载
curl -L -o model/silero_vad/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/site-packages/silero_vad/data/silero_vad.onnx
```

**SenseVoice BPE tokenizer 缺失：**
```bash
# 检查 BPE 模型文件是否存在
ls model/SenseVoiceSmall-onnx/chn_jpn_yue_eng_ko_spectok.bpe.model

# 补下载
curl -L -o model/SenseVoiceSmall-onnx/chn_jpn_yue_eng_ko_spectok.bpe.model \
  https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/chn_jpn_yue_eng_ko_spectok.bpe.model
```

### 其他文件缺失

检查模型目录，对比以下必需文件列表：

| 模型 | 必需文件 |
|------|----------|
| Silero VAD | `silero_vad.onnx` |
| SenseVoiceSmall ONNX | `model_quant.onnx`, `chn_jpn_yue_eng_ko_spectok.bpe.model`, `am.mvn`, `config.yaml`, `tokens.json` |
| Hy-MT2-1.8B GGUF | `Hy-MT2-1.8B-1.25Bit.gguf` + STQ 版 llama.cpp |
