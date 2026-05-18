@echo off
setlocal enabledelayedexpansion

set "NAME=csv_plot_pyqt6"
set "ENTRY=csv_plot_pyqt6.py"

echo Building %NAME%...
echo Entry file: %ENTRY%

nuitka --standalone ^
  --output-filename=%NAME% ^
  --enable-plugin=pyqt6 ^
  --enable-plugin=numpy ^
  --include-package=src ^
  --include-package=asammdf ^
  --include-module=src.data.mdf_loader ^
  --include-module=numpy ^
  --include-module=pandas ^
  --include-module=chardet ^
  --include-module=charset_normalizer ^
  --include-module=lxml ^
  --include-module=scipy ^
  --include-data-file=assets/icon.ico=assets/icon.ico ^
  --include-data-file=assets/icon.icns=assets/icon.icns ^
  --include-data-file=assets/icon.png=assets/icon.png ^
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
