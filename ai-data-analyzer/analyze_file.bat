@echo off
echo 🤖 AI Data Analyzer - Quick File Analysis
echo ===============================================
echo.

REM Get file path from user
set /p file_path="📁 Drag and drop your file here (or type path): "

REM Remove quotes if present
set file_path=%file_path:"=%

REM Check if file exists
if not exist "%file_path%" (
    echo ❌ File not found: %file_path%
    pause
    exit /b 1
)

echo.
echo 💬 What would you like to analyze? Examples:
echo    • Describe this data
echo    • Find correlations between all variables  
echo    • Predict [column name] using other features
echo    • Find customer segments
echo    • Detect anomalies
echo.

set /p prompt="🎯 Your analysis request: "

if "%prompt%"=="" set prompt=Describe this data

echo.
echo 🚀 Running analysis...
echo 📋 File: %file_path%
echo 📋 Prompt: %prompt%
echo ===============================================

REM Navigate to analyzer directory and run
cd /d "C:\Users\Elite\ai-data-analyzer"
python src\ai_data_analyzer.py --data "%file_path%" --prompt "%prompt%"

echo.
echo ✅ Analysis complete! Check the outputs folder for results.
pause
