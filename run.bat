@echo off
cd /d "%~dp0"

set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PYEXE%" (
  set "PYEXE=python"
)

rem Если Ollama уже запущена - не стартуем вторую копию
curl -s -m 2 http://localhost:11434/api/version >nul 2>nul
if errorlevel 1 (
  start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
  timeout /t 3 /nobreak >nul
)

"%PYEXE%" main.py
if errorlevel 1 (
  echo.
  echo ==== Oshibka zapuska, sm. tekst vyshe ====
  pause
)
