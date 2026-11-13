# Installing nn2excalidraw for OpenCode

There is one skill. Symlink `skills/nn2excalidraw` to `~/.config/opencode/skills/nn2excalidraw`.

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Python 3.10+ (for the CLI)

## Installation

### macOS / Linux

```bash
git clone https://github.com/bot66/nn2excalidraw.git ~/.nn2excalidraw

mkdir -p ~/.config/opencode/skills
ln -s ~/.nn2excalidraw/skills/nn2excalidraw ~/.config/opencode/skills/nn2excalidraw

pip install -e "$HOME/.nn2excalidraw[dev]"
```

Tracing with PyTorch:

```bash
pip install -e "$HOME/.nn2excalidraw[trace,dev]"
```

### Windows (PowerShell)

```powershell
git clone https://github.com/bot66/nn2excalidraw.git "$env:USERPROFILE\.nn2excalidraw"

New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.config\opencode\skills"
New-Item -ItemType SymbolicLink -Path "$env:USERPROFILE\.config\opencode\skills\nn2excalidraw" -Target "$env:USERPROFILE\.nn2excalidraw\skills\nn2excalidraw"

pip install -e "$env:USERPROFILE\.nn2excalidraw[dev]"
```

> **Note:** Creating symbolic links on Windows may require administrator privileges or Developer Mode enabled.

Restart OpenCode to discover the skill.

Verify by asking: "List available skills". `nn2excalidraw --help` should also run.

## Available skill

- **nn2excalidraw** — Turn a PyTorch `nn.Module` into a deterministic Excalidraw architecture diagram (ResNet-18 and ViT-B/16 are the reference models)

## Updating

```bash
cd ~/.nn2excalidraw && git pull
```

The symlink keeps pointing at the clone. You do not need to re-link.

## Uninstalling

### macOS / Linux

```bash
rm -f ~/.config/opencode/skills/nn2excalidraw
rm -rf ~/.nn2excalidraw
```

### Windows (PowerShell)

```powershell
Remove-Item -Force "$env:USERPROFILE\.config\opencode\skills\nn2excalidraw"
Remove-Item -Recurse -Force "$env:USERPROFILE\.nn2excalidraw"
```

## Troubleshooting

### Skill not found

1. Verify the symlink exists: `ls -la ~/.config/opencode/skills/`
2. `~/.config/opencode/skills/nn2excalidraw/SKILL.md` should exist
3. Restart OpenCode after installation
4. If the diagram command is missing, install the CLI from the clone root (`pip install -e ".[dev]"`)

## Getting help

- Issues: <https://github.com/bot66/nn2excalidraw/issues>
