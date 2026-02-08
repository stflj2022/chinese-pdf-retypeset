# -*- mode: python ; coding: utf-8 -*-

import os

# 图标路径（不同平台使用不同格式）
icon_path = None
if os.name == 'nt':  # Windows
    icon_path = 'icon.ico'  # 需要.ico格式
elif os.name == 'posix':  # Linux/macOS
    icon_path = 'icon.png'  # 需要.png格式

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src'), ('icon.png', '.') if os.path.exists('icon.png') else []],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='chinese-pdf-retypeset',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path if icon_path else None,
)
