@echo off
setlocal enabledelayedexpansion

set "SPEC=scripts\csv_plot_pyinstaller.spec"

echo =============================================
echo  Building csv_plot with PyInstaller (optimized)
echo =============================================
echo Spec file: %SPEC%
echo.

pyinstaller --noconfirm --clean "%SPEC%"

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
