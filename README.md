# 论文 PDF 翻译器

一个本地运行的 PDF 翻译工具：上传论文 PDF，调用可配置的官方或第三方翻译 API，并把译文写回原来的文本位置，尽量保持原 PDF 的页面、图片、表格和排版。

支持最大 50MB 的 PDF 文件。翻译过程中会显示进度条、当前阶段、已翻译行数和批次数；完成后可以直接下载，下载时也会显示下载进度。

## 加速与稳定性

- `并发数`：同时翻译多个文本批次。建议从 2 或 3 开始，过高可能触发 API 限速。
- `失败重试`：单个批次失败后自动重试，适合网络波动或模型临时繁忙。
- `页码范围`：可选 `全部页面`、`前 N 页` 或 `自定义`。自定义支持 `1-5,8,10-12`、中文逗号和 `1～5`；后端也兼容 `全部`、`前5页`。
- `API 池`：可同时使用多组 API。每行一组，格式为 `类型|URL|Key|模型`，例如 `gemini|https://generativelanguage.googleapis.com/v1beta/openai|AIza...|gemini-2.5-flash`。批次会轮询分配到不同 API，某组失败后会按重试设置切到下一组。
- `翻译模式`：`快速` 保持最高吞吐；`精准保护` 会在翻译前保护 URL、DOI、引用、公式和图表编号，翻译后再还原，适合论文。
- `排版方式`：`段落块` 是默认模式，会按 PDF 文本块合并翻译并整块回填，更接近 BabelDOC 的段落级处理；`逐行兼容` 保留旧的逐行替换逻辑。
- `译文字体`：可选择自动、宋体、黑体、Arial Unicode 或内置 Helvetica；译文会继续使用原 PDF 对应文本的颜色。
- 图表/公式保护：表格、图注、公式、页眉页脚会自动保留原文和版式，不作为普通段落翻译，避免图表页被挤乱。
- `翻译缓存`：相同文本、相同模型和相同模式下会复用译文，失败后重跑或重复内容较多时可减少 API 请求。
- `记住密钥和配置`：勾选后会把 API Key、API 池和常用设置保存到当前浏览器 `localStorage`，下次打开自动填回。可随时点 `清除已保存密钥` 删除。

## 运行

```bash
python3 app.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

## 支持的 API 类型

- `OpenAI-compatible`：适合 OpenAI 官方接口，以及兼容 `/v1/chat/completions` 的第三方 URL。
- `DeepSeek 官方 API`：默认使用 `https://api.deepseek.com/chat/completions`，模型默认 `deepseek-v4-flash`，也可选 `deepseek-v4-pro`、`deepseek-chat`、`deepseek-reasoner`。
- `Gemini 官方 API`：使用 Google 官方 OpenAI-compatible 入口 `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`，模型默认 `gemini-2.5-flash`。填写 Google AI Studio 获取的 Gemini API Key 即可，不需要反代 Gemini 网页端。
- `DeepL`：适合 DeepL `/v2/translate`。
- `LibreTranslate`：适合 LibreTranslate `/translate`。
- `Custom JSON`：向第三方 URL 发送 `{ "text": "...", "source": "...", "target": "..." }`，并尝试读取常见返回字段。
- `Mock`：不调用外部 API，只在每行前加目标语言标记，用于快速测试版式替换。

页面提供两个 API 辅助操作：

- `拉取模型`：OpenAI-compatible、DeepSeek 和 Gemini 模式会根据 API URL 推导模型列表接口并读取模型列表。
- `测试连接`：使用当前 API URL、API Key 和模型发送一次轻量检查，确认接口可用。

## 版式说明

默认的 `段落块` 模式会读取 PDF 文本块的位置、字号和颜色，先按段落合并翻译，再遮盖原段落并把译文写回同一区域。译文可使用指定中文字体，并会保留原文字体颜色。图片、线条和大多数页面结构会保留。`逐行兼容` 模式仍可用于文本块识别异常的 PDF。复杂 PDF 可能存在以下限制：

- 译文比原文长很多时，会自动缩小字号并截在原文本框内。
- 扫描版 PDF 只有图片，没有可提取文字，需要先 OCR。
- 原 PDF 的特殊字体、公式、脚注密集布局可能需要人工微调。
