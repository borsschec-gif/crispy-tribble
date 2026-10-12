@echo off
REM Запуск из исходников (нужен Python). Без Python скачай autotune.exe из Releases.
pip install -r requirements.txt >nul 2>&1
python autotune.py --speed 0.9
pause
