@echo off
cd /d "%~dp0"
set PYTHON_EXE=D:\Anaconda\envs\pytorch_learning\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python
"%PYTHON_EXE%" main.py prepare
pause
