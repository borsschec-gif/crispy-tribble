@echo off
REM Автотюн -> VB-Audio Virtual Cable. Поменяй параметры под себя.
pip install -r requirements.txt >nul 2>&1
python autotune.py -o "CABLE Input" -s chromatic --speed 0.9
pause
