@echo off
REM Script để retrain model (cho Windows Task Scheduler)
cd /d D:\HCMUTE_Nam4\TLCN\CodeSphere_ML
python src/training/auto_retrain.py
pause

