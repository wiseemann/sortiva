@echo off
cd /d "%~dp0"

echo =========================================================
echo   Smart Waste Management System  ^|  Adiwiyata
echo =========================================================

:: Activate virtual environment if present
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)

:: Launch the Streamlit app
echo Starting app...
python -m streamlit run app.py
pause
