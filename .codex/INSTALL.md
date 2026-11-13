# Installing nn2excalidraw for Codex

Enable the skill in Codex via native skill discovery. Clone the repo and symlink its `skills` directory into `~/.agents/skills`.

## Prerequisites

- Git
- Python 3.10+ (for the CLI)

## Installation

### macOS / Linux

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bot66/nn2excalidraw.git ~/.codex/nn2excalidraw
   ```

2. **Create the skills symlink:**

   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/nn2excalidraw/skills ~/.agents/skills/nn2excalidraw
   ```

3. **Install the CLI** from the clone root. The skill runs `nn2excalidraw`.

   ```bash
   pip install -e "$HOME/.codex/nn2excalidraw[dev]"
   ```

   Tracing with PyTorch:

   ```bash
   pip install -e "$HOME/.codex/nn2excalidraw[trace,dev]"
   ```

4. **Restart Codex** (quit and relaunch the CLI) to discover the skill.

### Windows (PowerShell)

```powershell
git clone https://github.com/bot66/nn2excalidraw.git "$env:USERPROFILE\.codex\nn2excalidraw"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
cmd /c mklink /J "$env:USERPROFILE\.agents\skills\nn2excalidraw" "$env:USERPROFILE\.codex\nn2excalidraw\skills"
pip install -e "$env:USERPROFILE\.codex\nn2excalidraw[dev]"
```

Tracing with PyTorch:

```powershell
pip install -e "$env:USERPROFILE\.codex\nn2excalidraw[trace,dev]"
```

Restart Codex (quit and relaunch the CLI) to discover the skill.

## Available skill

- **nn2excalidraw** — Turn a PyTorch `nn.Module` into a deterministic Excalidraw architecture diagram (ResNet-18 and ViT-B/16 are the reference models)

## Verify

### macOS / Linux

```bash
ls -la ~/.agents/skills/nn2excalidraw
nn2excalidraw --help
```

You should see a symlink pointing at `~/.codex/nn2excalidraw/skills`.

### Windows (PowerShell)

```powershell
Get-Item "$env:USERPROFILE\.agents\skills\nn2excalidraw"
nn2excalidraw --help
```

You should see a junction pointing at the clone's `skills` directory.

## Updating

### macOS / Linux

```bash
cd ~/.codex/nn2excalidraw && git pull
```

The symlink follows the clone, so you do not re-link.

### Windows (PowerShell)

```powershell
Set-Location "$env:USERPROFILE\.codex\nn2excalidraw"
git pull
```

## Uninstalling

### macOS / Linux

```bash
rm ~/.agents/skills/nn2excalidraw
```

Optionally delete the clone: `rm -rf ~/.codex/nn2excalidraw`.

### Windows (PowerShell)

```powershell
cmd /c rmdir "$env:USERPROFILE\.agents\skills\nn2excalidraw"
Remove-Item -Recurse -Force "$env:USERPROFILE\.codex\nn2excalidraw"
```

`rmdir` removes the junction without deleting the files it points at. The second command deletes the clone.
