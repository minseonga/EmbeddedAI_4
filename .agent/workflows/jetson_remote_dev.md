---
description: How to set up VS Code Remote SSH for Jetson Nano development
---

# Jetson Nano Remote Development Workflow

Instead of manually copying files and running commands on the Jetson, you can connect your VS Code directly to the device. This makes the Jetson feel like it's running locally on your Mac.

## 1. Prerequisites (On Mac)

1.  **Install VS Code Extension**:
    - Open VS Code.
    - Go to Extensions (`Cmd+Shift+X`).
    - Search for and install **"Remote - SSH"** (by Microsoft).

## 2. Connect to Jetson

1.  **Open Command Palette**: Press `Cmd+Shift+P`.
2.  Type **"Remote-SSH: Connect to Host..."** and select it.
3.  **Enter Connection String**:
    - Format: `username@ip-address`
    - Example: `jetson@192.168.0.15` (Check IP on Jetson with `ifconfig`).
4.  **Enter Password**: When prompted, type your Jetson's password.

## 3. Working Remotely

Once connected, a new VS Code window will open.
- **File Explorer**: Shows files *on the Jetson*, not your Mac. You can edit `app.py`, `pipeline.py` directly here.
- **Terminal** (`Ctrl+~`): This is a terminal *actually running on the Jetson*.
    - You can run `python benchmark_optimization.py` directly here.
    - No need to copy files back and forth!

## 4. Why this is better?

- **Real Hardware**: You are running on the actual Jetson GPU/CPU, so benchmarks are accurate.
- **Local Comfort**: You use your Mac's keyboard, screen, and IDE shortcuts.
- **No Emulation Bugs**: Simulating Jetson on Mac is nearly impossible due to GPU differences (CUDA vs Metal). This workflow avoids that entirely.
