@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem  HCMUT Thesis Template PDF Build Script (Windows)
rem  Compiles the LaTeX thesis in the official school format
rem  with proper bibliography handling.
rem
rem  Usage:
rem    build_template_thesis.bat [--clean] [--output <dir>]
rem      --clean, -c     Remove auxiliary files before building
rem      --output, -o    Copy the generated PDF to <dir>
rem ============================================================

rem Make sure we run from the script's directory
pushd "%~dp0"

rem ---------- Configuration ----------
set "LATEX_DIR=report\HCMUT_Master_Thesis_Template"
set "MAIN_FILE=main"
set "TARGET_NAME=2370116_LePhucDuc_ThesisReport_HCMUT"
set "OUTPUT_DIR=output"

rem ---------- Parse command line arguments ----------
set "CLEAN_FIRST=0"
set "OUTPUT_ARG="

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--clean" (
    set "CLEAN_FIRST=1"
    shift
    goto :parse_args
)
if "%~1"=="-c" (
    set "CLEAN_FIRST=1"
    shift
    goto :parse_args
)
if "%~1"=="--output" (
    set "OUTPUT_ARG=%~2"
    if "!OUTPUT_ARG!"=="" set "OUTPUT_ARG=%OUTPUT_DIR%"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-o" (
    set "OUTPUT_ARG=%~2"
    if "!OUTPUT_ARG!"=="" set "OUTPUT_ARG=%OUTPUT_DIR%"
    shift
    shift
    goto :parse_args
)
echo [ERROR] Unknown option: %~1
exit /b 1
:args_done

rem ---------- Check working directory ----------
if not exist "%LATEX_DIR%" (
    echo [ERROR] LaTeX directory "%LATEX_DIR%" not found!
    exit /b 1
)

rem ---------- Check dependencies ----------
call :check_dependencies
if errorlevel 1 exit /b 1

rem ---------- Clean auxiliary files if requested ----------
if "%CLEAN_FIRST%"=="1" (
    call :clean_aux_files
    if errorlevel 1 exit /b 1
)

rem ---------- Build ----------
call :compile_latex
if errorlevel 1 exit /b 1

rem ---------- Check output and copy if needed ----------
call :check_output
if errorlevel 1 exit /b 1

rem ---------- Show statistics ----------
call :show_stats

echo.
echo [SUCCESS] Build completed successfully!
exit /b 0

rem ============================================================
rem  Subroutines
rem ============================================================

:check_dependencies
echo [INFO] Checking LaTeX dependencies...
where pdflatex >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pdflatex not found. Please install TeX Live or MiKTeX.
    exit /b 1
)
where bibtex >nul 2>&1
if errorlevel 1 (
    echo [ERROR] bibtex not found. Please install TeX Live or MiKTeX.
    exit /b 1
)
echo [SUCCESS] All dependencies found.
exit /b 0

:clean_aux_files
echo [INFO] Cleaning auxiliary files...
pushd "%LATEX_DIR%"
del /q "*.aux" "*.bbl" "*.blg" "*.fdb_latexmk" "*.fls" "*.log" "*.out" "*.synctex.gz" "*.toc" "*.lof" "*.lot" "*.nav" "*.snm" "*.vrb" 2>nul
del /q "chapters\*.aux" "ext_pages\*.aux" 2>nul
popd
echo [SUCCESS] Auxiliary files cleaned.
exit /b 0

:compile_latex
echo [INFO] Starting LaTeX compilation for HCMUT Thesis Template...
pushd "%LATEX_DIR%"

rem First pass
call :compile_pass 1
if errorlevel 1 ( popd & exit /b 1 )

rem Run bibtex if main.bib exists
if exist "%MAIN_FILE%.bib" (
    echo [INFO] Running bibtex...
    bibtex "%TARGET_NAME%" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] BibTeX failed, but continuing...
    ) else (
        echo [SUCCESS] BibTeX completed successfully.
    )
)

rem Second pass
call :compile_pass 2
if errorlevel 1 ( popd & exit /b 1 )

rem Third pass (to resolve all cross-references)
call :compile_pass 3
if errorlevel 1 ( popd & exit /b 1 )

popd
echo [SUCCESS] LaTeX compilation completed successfully.
exit /b 0

:compile_pass
set "PASS_LABEL=%~1"
if "%PASS_LABEL%"=="1" set "PASS_LABEL=1st"
if "%PASS_LABEL%"=="2" set "PASS_LABEL=2nd"
if "%PASS_LABEL%"=="3" set "PASS_LABEL=3rd"
echo [INFO] Running pdflatex (%PASS_LABEL% pass)...
pdflatex -interaction=nonstopmode -jobname="%TARGET_NAME%" "%MAIN_FILE%.tex" > "pdflatex_%PASS_LABEL%.log" 2>&1
if errorlevel 1 (
    findstr /C:"Output written on %TARGET_NAME%.pdf" "%TARGET_NAME%.log" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] pdflatex %PASS_LABEL% pass failed. Check %TARGET_NAME%.log for details.
        powershell -NoProfile -Command "Get-Content -LiteralPath '%TARGET_NAME%.log' -Tail 20"
        exit /b 1
    )
)
exit /b 0

:check_output
if not exist "%LATEX_DIR%\%TARGET_NAME%.pdf" (
    echo [ERROR] PDF generation failed!
    exit /b 1
)
for %%A in ("%LATEX_DIR%\%TARGET_NAME%.pdf") do set "FILESIZE=%%~zA"
set /a SIZE_MB=!FILESIZE! / 1048576
echo [SUCCESS] PDF generated successfully: %LATEX_DIR%\%TARGET_NAME%.pdf (%SIZE_MB% MB)
if not "%OUTPUT_ARG%"=="" (
    if not exist "%OUTPUT_ARG%" mkdir "%OUTPUT_ARG%"
    copy /y "%LATEX_DIR%\%TARGET_NAME%.pdf" "%OUTPUT_ARG%\" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to copy PDF to %OUTPUT_ARG%
        exit /b 1
    )
    echo [SUCCESS] PDF copied to: %OUTPUT_ARG%\%TARGET_NAME%.pdf
)
exit /b 0

:show_stats
if not exist "%LATEX_DIR%\%TARGET_NAME%.log" exit /b 0
echo [INFO] Compilation Statistics:
powershell -NoProfile -Command "$log = '%LATEX_DIR%\%TARGET_NAME%.log'; $m = Select-String -LiteralPath $log -Pattern 'Output written on .*\((\d+) pages' | Select-Object -Last 1; $pages = 'Unknown'; if ($m) { $pages = $m.Matches[0].Groups[1].Value }; $warn = (Select-String -LiteralPath $log -Pattern 'Warning').Count; $err = (Select-String -LiteralPath $log -Pattern 'Error').Count; Write-Host ('  Pages: ' + $pages); Write-Host ('  Warnings: ' + $warn); Write-Host ('  Errors: ' + $err); if ($warn -gt 0) { Write-Host '[WARNING] There were warnings. Check the .log for details.' }"
exit /b 0
