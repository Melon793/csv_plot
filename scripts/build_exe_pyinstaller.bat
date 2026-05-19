@echo off
setlocal enabledelayedexpansion

set "NAME=csv_plot_pyqt6"
set "ENTRY=csv_plot_pyqt6.py"
set "ICON=assets/icon.ico"

echo Building %NAME% with PyInstaller...
echo Entry file: %ENTRY%
echo Icon: %ICON%

pyinstaller --noconfirm ^
  --onedir ^
  --windowed ^
  --name=%NAME% ^
  --icon="%ICON%" ^
  --add-data "assets;assets" ^
  --add-data "README.md;." ^
  --hidden-import=asammdf ^
  --hidden-import=chardet ^
  --hidden-import=charset_normalizer ^
  --hidden-import=src ^
  --collect-submodules numpy ^
  --collect-submodules pandas ^
  --collect-submodules PyQt6 ^
  --exclude-module=nuitka ^
  --exclude-module=pytest ^
  --exclude-module=pyinstaller ^
  --exclude-module=tkinter ^
  --noupx ^
  --strip ^
  --upx-exclude=vcruntime140.dll ^
  --upx-exclude=python3.dll ^
  --noupx ^
  "%ENTRY%"

if %ERRORLEVEL% EQU 0 (
  echo Building successfully!
  echo Executable is in: dist\%NAME%\目录下
) else (
  echo Building failed!
  pause
)