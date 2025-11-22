@echo off
echo ========================================
echo   DETECTION DE CERCLE DE FIBRE OPTIQUE
echo ========================================
echo.

REM Verifier si Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    echo Veuillez installer Python depuis python.org
    pause
    exit /b 1
)

REM Verifier si OpenCV est installe
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installation d'OpenCV...
    pip install opencv-python numpy
)

REM Verifier les arguments
if "%~1"=="" (
    echo [ERREUR] Veuillez specifier une image a analyser
    echo.
    echo Usage: detect_fiber.bat image.png
    echo.
    pause
    exit /b 1
)

REM Executer le script Python
echo [INFO] Analyse de l'image: %1
echo.

python fiber_circle_detection_windows.py "%~1" --no-show

echo.
echo ========================================
echo   TRAITEMENT TERMINE
echo ========================================
pause
