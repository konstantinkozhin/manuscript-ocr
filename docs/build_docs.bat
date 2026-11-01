@ECHO OFF
REM Build documentation script for manuscript-ocr
REM Builds HTML documentation for English and Russian versions

SETLOCAL

SET SCRIPT_DIR=%~dp0
SET SPHINX_DIR=%SCRIPT_DIR%sphinx
SET BUILD_DIR=%SPHINX_DIR%\_build
SET SPHINX_BUILD=%SCRIPT_DIR%..\env\Scripts\python.exe
SET DOCS_VERSION=0.1.13
SET SPHINX_INTL=%SCRIPT_DIR%..\env\Scripts\sphinx-intl.exe

ECHO ====================================
ECHO Building manuscript-ocr Documentation
ECHO ====================================
ECHO.

REM Check if virtual environment exists
IF NOT EXIST "%SPHINX_BUILD%" (
    ECHO ERROR: Virtual environment not found or sphinx not installed
    ECHO Please run: pip install sphinx sphinx-rtd-theme numpydoc sphinx-autodoc-typehints sphinxcontrib-mermaid sphinx-intl
    EXIT /B 1
)

REM Clean old build
IF EXIST "%BUILD_DIR%\html" (
    ECHO Cleaning old build directory...
    RMDIR /S /Q "%BUILD_DIR%\html"
    ECHO Done.
    ECHO.
)

CD /D "%SPHINX_DIR%"

REM Update .pot files and .po files
ECHO Updating translation templates...
"%SPHINX_BUILD%" -m sphinx -b gettext -D release=%DOCS_VERSION% -D version=%DOCS_VERSION% . _build/gettext
"%SPHINX_INTL%" update -p _build/gettext -l ru
ECHO Done.
ECHO.

REM Build English documentation
ECHO Building English documentation...
"%SPHINX_BUILD%" -m sphinx -b html -D release=%DOCS_VERSION% -D version=%DOCS_VERSION% . _build/html/en
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: English build failed!
    EXIT /B 1
)
ECHO Done.
ECHO.

REM Build Russian documentation
ECHO Building Russian documentation...
"%SPHINX_BUILD%" -m sphinx -b html -D release=%DOCS_VERSION% -D version=%DOCS_VERSION% -D language=ru . _build/html/ru
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Russian build failed!
    EXIT /B 1
)
ECHO Done.
ECHO.

REM Create redirect index.html at root
ECHO Creating root redirect...
(
    ECHO ^<!DOCTYPE html^>
    ECHO ^<html^>
    ECHO ^<head^>
    ECHO     ^<meta charset="utf-8"^>
    ECHO     ^<meta http-equiv="refresh" content="0; url=en/index.html"^>
    ECHO     ^<title^>Redirecting...^</title^>
    ECHO ^</head^>
    ECHO ^<body^>
    ECHO     ^<p^>Redirecting to ^<a href="en/index.html"^>English documentation^</a^>...^</p^>
    ECHO ^</body^>
    ECHO ^</html^>
) > "%BUILD_DIR%\html\index.html"
ECHO Done.
ECHO.

ECHO.
ECHO ====================================
ECHO Build completed successfully!
ECHO ====================================
ECHO.
ECHO Documentation is available at:
ECHO   English: %BUILD_DIR%\html\en\index.html
ECHO   Russian: %BUILD_DIR%\html\ru\index.html
ECHO.
ECHO Opening English version in browser...
start "" "%BUILD_DIR%\html\en\index.html"

ENDLOCAL
EXIT /B 0
