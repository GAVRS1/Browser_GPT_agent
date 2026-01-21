@echo off
echo Opening Playwright Chromium with agent profile...

set CHROME_PATH=%USERPROFILE%\AppData\Local\ms-playwright\chromium-1194\chrome-win\chrome.exe
set PROFILE_DIR=%CD%\user_data

"%CHROME_PATH%" --user-data-dir="%PROFILE_DIR%"

pause
