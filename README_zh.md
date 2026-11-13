# nn2excalidraw

[English](./README.md)

把 PyTorch `nn.Module` 转成确定性的 Excalidraw 架构图。

入口类和详细程度由你选择。布局引擎根据结构化图摆放每一个框，同一份代码永远得到同一张图。默认是块级视图：ResNet 的 `BasicBlock` 是一张带残差弧的卡片，而不是一堆 Conv/BN/ReLU。ViT 画出 patch embed、class token，以及一个标了 × 12 的展开 Transformer 块。

不需要 GPU，不下载预训练权重，运行时也不访问 excalidraw.com。

## 一分钟

```bash
pip install -e ".[dev]"

nn2excalidraw examples/resnet18.py --class ResNet \
  --input-shape 1,3,224,224 \
  -o examples/outputs/resnet18.excalidraw

nn2excalidraw examples/vit_b_16.py --class VisionTransformer \
  --input-shape 1,3,224,224 \
  -o examples/outputs/vit_b_16.excalidraw
```

在 [excalidraw.com](https://excalidraw.com) 用 **Open** 打开文件。CLI 会打印路径和图摘要：

```text
wrote /…/resnet18.excalidraw
entry: ResNet  mode: static  detail: block  nodes: 13  edges: 20  residual: 8  skip: 0
```

仓库里已经放好的图，含 Obsidian 包装：

- [`examples/outputs/resnet18.excalidraw`](examples/outputs/resnet18.excalidraw)
- [`examples/outputs/resnet18.excalidraw.md`](examples/outputs/resnet18.excalidraw.md)
- [`examples/outputs/vit_b_16.excalidraw`](examples/outputs/vit_b_16.excalidraw)
- [`examples/outputs/vit_b_16.excalidraw.md`](examples/outputs/vit_b_16.excalidraw.md)

`examples/resnet18.py` 对应 torchvision 的 `resnet18`（BasicBlock，`[2, 2, 2, 2]`，通道 64-128-256-512）。`examples/vit_b_16.py` 对应 torchvision 的 `vit_b_16`（patch 16，隐藏维 768，MLP 3072，12 头，12 块，class token）。两者都写在仓库里，不会下载权重。

## CLI

```text
nn2excalidraw PATH
  --class NAME
  --input-shape 1,3,224,224
  --detail block|full          # 默认 block
  --format excalidraw|obsidian # 默认 excalidraw
  --mode auto|static|trace     # 默认 auto
  --direction ltr|ttb          # 默认 ltr
  -o FILE
  --title TEXT
  --dump-ir
```

| 参数 | 含义 |
| --- | --- |
| `--class` | 入口 `nn.Module`。文件里有多个根类时必须指定 |
| `--input-shape` | 用于 `B×64×56×56` 这类标注的输入形状。不知道就不要编 |
| `--detail block` | 把 Conv/BN/ReLU 收成 ResidualBlock、MLP、Stem、Head、Attention |
| `--detail full` | 把这些块展开成内部调用 |
| `--format obsidian` | 写成 Obsidian Excalidraw 插件用的 `Title.excalidraw.md` |
| `--mode static` | 只解析，不导入文件 |
| `--mode trace` | 导入、构造，并可选地跑一次 forward 记录形状 |
| `--mode auto` | 能导入 torch 就追踪，否则静态解析 |

```python
from nn2excalidraw import diagram_from_path

diagram = diagram_from_path(
    "model.py",
    class_name="ResNet",
    input_shape=(1, 3, 224, 224),
    mode="static",
)
diagram.write("out.excalidraw")
diagram.write("out.excalidraw.md", fmt="obsidian")
print(diagram.summary)
```

## 图里有什么

**ResNet-18。** Stem（`7×7 s2`）、max-pool，然后 `layer1`–`layer4` 框里的八张 `BasicBlock`。每张卡有虚线残差弧和 `+`。`layer2`、`layer3`、`layer4` 的第一块标 `s2 · proj 1×1`。最后是全局池化和 `fc`。输入 `1×3×224×224` 时，layer1 为 `B×64×56×56`，到 layer4 为 `B×512×7×7`。

**ViT-B/16。** Patch 卷积 `16×16` → `B×196×768`，class token 拼到 `B×197×768`，加上位置编码，然后一个框，标题是 **EncoderBlock × 12**：LayerNorm、多头注意力（12 头，QKV）、残差加、MLP `768→3072→768`、再一次残差加。最后是 LayerNorm、class token 读出和线性头。

测试里还有：`examples/mlp.py`、`examples/sequential_cnn.py`、`examples/unet_tiny.py`（编码器-解码器跳跃连接）、`examples/tiny_transformer.py`。

颜色和布局见 [docs/visual-language.md](docs/visual-language.md)。流水线和静态解析能覆盖的 Python 子集见 [docs/architecture.md](docs/architecture.md)。

## 安装

本仓库是**私有**的。`git clone` 和 `claude plugin marketplace add` 都需要能访问 <https://github.com/bot66/nn2excalidraw>（带凭据的 HTTPS，或 SSH）。不要把 token 写进仓库、可分享的 shell 历史或对话里。

在 [PR #1](https://github.com/bot66/nn2excalidraw/pull/1) 合并之前，clone 时加上 `-b cursor/nn2excalidraw-v1-c1e1`。合并进 `main` 之后去掉 `-b`。

唯一的技能文件是 [`skills/nn2excalidraw/SKILL.md`](skills/nn2excalidraw/SKILL.md)。工具应指向 `skills/` 目录。技能会运行 `nn2excalidraw` 命令，所以还要在克隆根目录安装 CLI（见下文）。

### Claude Code

`claude plugin marketplace add` 读取默认分支，PR 合并之后才能装到这个技能。在那之前，用下面 Cursor、Codex 或 OpenCode 的方式克隆分支。

```bash
claude plugin marketplace add https://github.com/bot66/nn2excalidraw
claude plugin install nn2excalidraw
```

清单：[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json)、[`.claude-plugin/plugin.json`](.claude-plugin/plugin.json)。

### Cursor

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.cursor/nn2excalidraw
```

把 Cursor 的 skills 路径指到 `~/.cursor/nn2excalidraw/skills/`。

Windows 的安装、校验、更新和卸载见 [`.cursor-plugin/INSTALL.md`](.cursor-plugin/INSTALL.md)。

### Codex

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.codex/nn2excalidraw
mkdir -p ~/.agents/skills
ln -s ~/.codex/nn2excalidraw/skills ~/.agents/skills/nn2excalidraw
```

重启 Codex。Windows 说明见 [`.codex/INSTALL.md`](.codex/INSTALL.md)。

### OpenCode

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.nn2excalidraw
mkdir -p ~/.config/opencode/skills
ln -s ~/.nn2excalidraw/skills/nn2excalidraw ~/.config/opencode/skills/nn2excalidraw
```

重启 OpenCode。详见 [`.opencode/INSTALL.md`](.opencode/INSTALL.md)。

### Python 包（CLI）

在克隆根目录（`~/.cursor/nn2excalidraw`、`~/.codex/nn2excalidraw` 或 `~/.nn2excalidraw`）：

```bash
pip install -e ".[dev]"          # 静态解析和 CLI
pip install -e ".[trace,dev]"    # 再加上 PyTorch 追踪
```

需要 Python 3.10+。Torch 可选。追踪在 CPU 上做一次 dummy forward。不装这个包的话，智能体看得到技能，却跑不了 `nn2excalidraw`。

### VS Code

本仓库不提供独立的 VS Code 扩展。

在集成终端里使用 Codex、Claude Code 或 OpenCode。如果要直接配置本仓库的本地 skills 路径，用 Cursor，并参考 [`.cursor-plugin/INSTALL.md`](.cursor-plugin/INSTALL.md)。

## 测试

```bash
pytest
```

覆盖图 IR（ResNet 残差、ViT 的重复块与 attention/MLP、U-Net 跳跃连接）、两种导出、确定性和 CLI。安装了 PyTorch 时 `tests/test_trace.py` 会检查 ResNet-18 和 ViT-B/16 的追踪结果与静态解析一致。

## 信任

静态模式不导入你的文件。追踪模式会：导入会执行顶层代码，dummy forward 会执行 `forward`。只给信任的代码画图。详见 [docs/trust-model.md](docs/trust-model.md)。

## 限制

- 只支持 PyTorch。v1 不做 TensorFlow 和 JAX。
- 静态解析覆盖常见模型写法（`__init__` 构建、`Sequential`、残差加、`torch.cat`、reshape/permute）。构造函数过于动态时用 `--mode trace`，或把模块写得更直白。
- 连续 4 个及以上相同的兄弟模块会收成一张（`EncoderBlock × 12`）。ResNet-18 每一段只有两块，所以都会画出来。
- 默认视图里，块内部的 BatchNorm 和激活不是单独的框。要看它们就加 `--detail full`。
- 设备、分布式包装和自定义 autograd 函数不单独处理。

## 许可证

MIT。附带的 yuelfei 深度学习 Excalidraw 库同样是 MIT，见 [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md)。运行时画图不会加载那个文件。
