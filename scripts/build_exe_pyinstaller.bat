@echo off
setlocal enabledelayedexpansion

rem 查找可用的 Python
where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set "PYTHON=uv run python"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

set "SPEC=scripts\csv_plot_pyinstaller.spec"

echo =============================================
echo  Building csv_plot with PyInstaller (optimized)
echo =============================================
echo Spec file: %SPEC%
echo.

echo Generating build info...
!PYTHON! scripts\generate_build_info.py
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo =============================================
  echo  Build info generation failed!
  echo =============================================
  pause
  exit /b 1
)
echo.

pyinstaller --noconfirm --clean "%SPEC%"

rem 构建结束后清理自动生成的构建信息文件，避免陈旧数据污染开发环境
if exist "src\_build_info.py" del /q "src\_build_info.py"

if %ERRORLEVEL% EQU 0 (
  echo.
  echo =============================================
  echo  Build succeeded!
  echo  Output: dist\csv_plot\
  echo =============================================
) else (
  echo.
  echo =============================================
  echo  Build failed!
  echo =============================================
  pause
  exit /b 1
)
