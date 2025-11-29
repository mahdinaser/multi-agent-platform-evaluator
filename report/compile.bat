@echo off
REM Batch script to generate IEEE paper PDF

echo ================================================
echo  Generating IEEE Conference Paper PDF
echo ================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.7+ from python.org
    echo.
    pause
    exit /b 1
)

echo Generating PDF using Python...
echo.

python generate_pdf.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: PDF generation failed!
    echo Check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo  PDF Generation Complete!
echo  Output: paper.pdf
echo ================================================
echo.

pause

