@echo off
echo ===========================================
echo Deepfake Audio Detection System Setup
echo ===========================================
echo Setup Virtual Environment...
if not exist "venv" (
    python.exe -m venv venv
)

echo Installing dependencies (this will bypass PowerShell restrictions)...
venv\Scripts\python.exe -m pip install -r requirements.txt

echo Fetching and mapping the Real Kaggle Dataset...
venv\Scripts\python.exe utils\setup_kaggle_dataset.py

echo Training the initial dummy convolutional model...
venv\Scripts\python.exe train.py

echo ===========================================
echo Launching the Streamlit Web Application
echo ===========================================
venv\Scripts\python.exe -m streamlit run app.py
pause
