@echo off
setlocal enabledelayedexpansion

set "NAME=csv_plot"
set "ENTRY=csv_plot.py"

echo ========================================
echo  Building %NAME% with Nuitka (optimized)
echo ========================================
echo Entry: %ENTRY%

nuitka --standalone ^
  --output-filename=%NAME% ^
  --enable-plugin=pyside6 ^
  --enable-plugin=numpy ^
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
  --nofollow-import-to=unittest ^
  --nofollow-import-to=test ^
  --nofollow-import-to=distutils ^
  --nofollow-import-to=setuptools ^
  --nofollow-import-to=pip ^
  --nofollow-import-to=wheel ^
  --nofollow-import-to=scipy ^
  --nofollow-import-to=lxml ^
  --nofollow-import-to=PIL ^
  --nofollow-import-to=cv2 ^
  --nofollow-import-to=matplotlib ^
  --lto=yes ^
  --jobs=8 ^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=assets/icon.ico ^
  %ENTRY%

if %ERRORLEVEL% EQU 0 (
  echo.
  echo ========================================
  echo  Build succeeded!
  echo  Output: %NAME%.dist\%NAME%.exe
  echo ========================================
) else (
  echo.
  echo ========================================
  echo  Build failed!
  echo ========================================
  pause
  exit /b 1
)
