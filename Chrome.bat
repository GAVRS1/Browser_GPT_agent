@echo off
echo Opening Playwright Chromium with agent profile...

set "PROFILE_DIR=%CD%\user_data"

for /f "delims=" %%P in ('where chrome 2^>nul') do set "CHROME_PATH=%%P" & goto :found
for /f "delims=" %%P in ('dir /b /s "%USERPROFILE%\AppData\Local\ms-playwright\chromium-*\\chrome-win\\chrome.exe" 2^>nul') do set "CHROME_PATH=%%P" & goto :found

echo Chrome executable not found. Install Chrome or Playwright Chromium.
pause
exit /b 1

:found
"%CHROME_PATH%" --user-data-dir="%PROFILE_DIR%"

pause
