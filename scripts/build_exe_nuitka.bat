@echo off
setlocal enabledelayedexpansion

set "NAME=csv_plot"
set "ENTRY=csv_plot.py"

echo Building %NAME%...
echo Entry file: %ENTRY%

nuitka --standalone ^
  --zig ^
  --output-filename=%NAME% ^
  --enable-plugin=pyside6 ^
  --include-package=src ^
  --include-module=asammdf ^
  --include-module=numpy ^
  --include-module=pandas ^
  --include-package=chardet ^
  --include-module=charset_normalizer ^
  --include-data-dir=assets=assets ^
  --include-data-file=README.md=README.md ^
  --follow-imports ^
  --nofollow-import-to=nuitka ^
  --nofollow-import-to=pytest ^
  --nofollow-import-to=pyinstaller ^
  --nofollow-import-to=tkinter ^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=assets/icon.ico ^
  %ENTRY%

if %ERRORLEVEL% EQU 0 (
  echo Building successfully!
) else (
  echo Building failed!
  pause
)