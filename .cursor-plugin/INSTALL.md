# Installing nn2excalidraw for Cursor

Enable the skill in Cursor by cloning the repository locally and pointing Cursor's skills path at the `skills/` directory.

This repository is private. Clone it with GitHub access (HTTPS credentials or SSH). Do not paste tokens into the repo or into chat.

Until [PR #1](https://github.com/bot66/nn2excalidraw/pull/1) is merged, clone with `-b cursor/nn2excalidraw-v1-c1e1`. After it lands on `main`, drop `-b`.

The skill runs the `nn2excalidraw` CLI. Install that from the clone root after the clone (see below).

## Prerequisites

- Cursor installed
- Git
- Python 3.10+ (for the CLI)

## Installation

### macOS / Linux

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.cursor/nn2excalidraw
pip install -e "$HOME/.cursor/nn2excalidraw[dev]"
```

Tracing with PyTorch:

```bash
pip install -e "$HOME/.cursor/nn2excalidraw[trace,dev]"
```

Set Cursor's skills path to:

```text
~/.cursor/nn2excalidraw/skills/
```

### Windows (PowerShell)

```powershell
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git "$env:USERPROFILE\.cursor\nn2excalidraw"
pip install -e "$env:USERPROFILE\.cursor\nn2excalidraw[dev]"
```

Tracing with PyTorch:

```powershell
pip install -e "$env:USERPROFILE\.cursor\nn2excalidraw[trace,dev]"
```

Set Cursor's skills path to:

```text
C:\Users\YOUR_USERNAME\.cursor\nn2excalidraw\skills\
```

Replace `YOUR_USERNAME` with your Windows account name.

After saving the path, restart Cursor or reload the window so it rescans local skills.

## Verify

Confirm the clone exists and contains `SKILL.md`:

### macOS / Linux

```bash
find ~/.cursor/nn2excalidraw/skills -maxdepth 2 -name SKILL.md | head
nn2excalidraw --help
```

### Windows (PowerShell)

```powershell
Get-ChildItem "$env:USERPROFILE\.cursor\nn2excalidraw\skills" -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -Filter SKILL.md
}
nn2excalidraw --help
```

## Updating

### macOS / Linux

```bash
cd ~/.cursor/nn2excalidraw && git pull
```

While the skill is only on `cursor/nn2excalidraw-v1-c1e1`, pull that branch (`git pull origin cursor/nn2excalidraw-v1-c1e1`). After merge, `git checkout main` and `git pull`.

### Windows (PowerShell)

```powershell
Set-Location "$env:USERPROFILE\.cursor\nn2excalidraw"
git pull
```

## Uninstalling

### macOS / Linux

```bash
rm -rf ~/.cursor/nn2excalidraw
```

### Windows (PowerShell)

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cursor\nn2excalidraw"
```

Remove the skills path from Cursor settings if you added it by hand.

## VS Code Note

This repository does not currently ship a standalone VS Code extension.

If you use VS Code, the recommended options are:

- run a supported CLI tool such as Codex, Claude Code, or OpenCode inside the VS Code integrated terminal
- use Cursor if you want native local-skills configuration from this repository
