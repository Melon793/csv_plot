@echo off
setlocal enabledelayedexpansion

set "NAME=csv_plot"
set "ENTRY=csv_plot.py"

echo ========================================
echo  Building %NAME% with Nuitka (optimized)
echo ========================================

rem === Includes / Plugins ===
rem === PySide6 排除: WebEngine / QML-Quick / 3D / Charts-Graphs / Multimedia / Hardware / 杂项 ===
rem === NumPy 排除: random / f2py / ma / polynomial / linalg / fft / testing ===
rem === Pandas 排除: tests / plotting ===

nuitka --standalone ^
  --output-filename=%NAME% ^
  --enable-plugin=pyside6 ^
  --enable-plugin=numpy ^
  --include-module=PySide6.QtOpenGL ^
  --include-module=PySide6.QtOpenGLWidgets ^
  --include-package=src ^
  --include-package=asammdf ^
  --include-package=asammdf.blocks ^
  --include-module=numpy ^
  --include-module=pandas ^
  --include-package=chardet ^
  --include-module=charset_normalizer ^
  --include-module=ujson ^
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
  --nofollow-import-to=PySide6.QtWebEngineWidgets ^
  --nofollow-import-to=PySide6.QtWebEngineCore ^
  --nofollow-import-to=PySide6.QtWebEngineQuick ^
  --nofollow-import-to=PySide6.QtWebChannel ^
  --nofollow-import-to=PySide6.QtWebSockets ^
  --nofollow-import-to=PySide6.QtQml ^
  --nofollow-import-to=PySide6.QtQuick ^
  --nofollow-import-to=PySide6.QtQuick3D ^
  --nofollow-import-to=PySide6.QtQuickControls2 ^
  --nofollow-import-to=PySide6.QtQuickWidgets ^
  --nofollow-import-to=PySide6.Qt3DAnimation ^
  --nofollow-import-to=PySide6.Qt3DCore ^
  --nofollow-import-to=PySide6.Qt3DExtras ^
  --nofollow-import-to=PySide6.Qt3DInput ^
  --nofollow-import-to=PySide6.Qt3DLogic ^
  --nofollow-import-to=PySide6.Qt3DRender ^
  --nofollow-import-to=PySide6.QtCharts ^
  --nofollow-import-to=PySide6.QtDataVisualization ^
  --nofollow-import-to=PySide6.QtGraphs ^
  --nofollow-import-to=PySide6.QtGraphsWidgets ^
  --nofollow-import-to=PySide6.QtMultimedia ^
  --nofollow-import-to=PySide6.QtMultimediaWidgets ^
  --nofollow-import-to=PySide6.QtSpatialAudio ^
  --nofollow-import-to=PySide6.QtBluetooth ^
  --nofollow-import-to=PySide6.QtNfc ^
  --nofollow-import-to=PySide6.QtSensors ^
  --nofollow-import-to=PySide6.QtSerialBus ^
  --nofollow-import-to=PySide6.QtSerialPort ^
  --nofollow-import-to=PySide6.QtPdf ^
  --nofollow-import-to=PySide6.QtPdfWidgets ^
  --nofollow-import-to=PySide6.QtHelp ^
  --nofollow-import-to=PySide6.QtLocation ^
  --nofollow-import-to=PySide6.QtPositioning ^
  --nofollow-import-to=PySide6.QtSql ^
  --nofollow-import-to=PySide6.QtDBus ^
  --nofollow-import-to=PySide6.QtRemoteObjects ^
  --nofollow-import-to=PySide6.QtScxml ^
  --nofollow-import-to=PySide6.QtStateMachine ^
  --nofollow-import-to=PySide6.QtTest ^
  --nofollow-import-to=PySide6.QtTextToSpeech ^
  --nofollow-import-to=PySide6.QtHttpServer ^
  --nofollow-import-to=PySide6.QtDesigner ^
  --nofollow-import-to=PySide6.QtUiTools ^
  --nofollow-import-to=numpy.random ^
  --nofollow-import-to=numpy.f2py ^
  --nofollow-import-to=numpy.ma ^
  --nofollow-import-to=numpy.polynomial ^
  --nofollow-import-to=numpy.linalg ^
  --nofollow-import-to=numpy.fft ^
  --nofollow-import-to=numpy.testing ^
  --nofollow-import-to=pandas.tests ^
  --nofollow-import-to=pandas.plotting ^
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
