@echo off
echo Starting Disaster Mission Control...
REM Using direct path to virtual environment python to avoid PATH issues
".\.venv\Scripts\python.exe" -m streamlit run disaster_mission_control.py
pause
