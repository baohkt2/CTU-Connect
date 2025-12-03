@echo off
@REM ----------------------------------------------------------------------------
@REM Licensed to the Apache Software Foundation (ASF) under one
@REM or more contributor license agreements.  See the NOTICE file
@REM distributed with this work for additional information
@REM regarding copyright ownership.  The ASF licenses this file
@REM to you under the Apache License, Version 2.0 (the
@REM "License"); you may not use this file except in compliance
@REM with the License.  You may obtain a copy of the License at
@REM
@REM    http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM Unless required by applicable law or agreed to in writing,
@REM software distributed under the License is distributed on an
@REM "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
@REM KIND, either express or implied.  See the License for the
@REM specific language governing permissions and limitations
@REM under the License.
@REM ----------------------------------------------------------------------------

@REM ----------------------------------------------------------------------------
@REM Apache Maven Wrapper startup batch script, version 3.3.2
@REM
@REM Optional ENV vars
@REM   MVNW_REPOURL - repo url base for downloading maven distribution
@REM   MVNW_USERNAME/MVNW_PASSWORD - user and password for downloading maven
@REM   MVNW_VERBOSE - true/false - enable verbose mode (default: false)
@REM ----------------------------------------------------------------------------

@REM Begin all REM lines with '@' in case MAVEN_BATCH_ECHO is 'on'
@echo off
@REM set title of command window
title %0
@REM enable echoing by setting MAVEN_BATCH_ECHO to 'on'
@if "%MAVEN_BATCH_ECHO%" == "on"  echo %MAVEN_BATCH_ECHO%

@REM set %HOME% to equivalent of $HOME
if "%HOME%" == "" (set "HOME=%HOMEDRIVE%%HOMEPATH%")

@REM ==== START VALIDATION ====
if not "%JAVA_HOME%" == "" (
    set "_JAVACMD=%JAVA_HOME%\bin\java.exe"
    set "_JAVACCMD=%JAVA_HOME%\bin\javac.exe"

    if not exist "%_JAVACMD%" (
        echo.
        echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
        echo.
        echo Please set the JAVA_HOME variable in your environment to match the
        echo location of your Java installation.
        echo.
        goto error
    )
else (
    where java >NUL 2>&1 && where javac >NUL 2>&1
    if not ERRORLEVEL 1 (
        set "_JAVACMD=java.exe"
        set "_JAVACCMD=javac.exe"
    ) else (
        echo.
        echo ERROR: The java/javac command does not exist in PATH or JAVA_HOME is not set
        echo.
        goto error
    )
)
@REM ==== END VALIDATION ====

for /f "usebackq tokens=1* delims=:" %%i in (`dir /s /b "%~dp0.mvn\wrapper\maven-wrapper.properties" 2^>NUL`) do set "wrapper_path=%%i:%%j"
for /f "tokens=1,2 delims==" %%a in (%wrapper_path%) do set "%%a=%%b"
if not "%distributionUrl%"=="" goto skipempty
echo Could not read maven-wrapper.properties^! >&2
exit /b 1
:skipempty

set "MAVEN_USER_HOME=%MAVEN_USER_HOME:~0,-1%"
if not "%MAVEN_USER_HOME%"=="" goto skipdefault
if not "%USERPROFILE%"=="" set "MAVEN_USER_HOME=%USERPROFILE%\.m2"
if not "%USERPROFILE%"=="" goto skipdefault
set "MAVEN_USER_HOME=%HOME%\.m2"
:skipdefault
if not exist "%MAVEN_USER_HOME%" mkdir "%MAVEN_USER_HOME%"

@REM Parse distributionUrl and optional distributionSha256Sum, already loaded from maven-wrapper.properties
@REM
@REM Expected format
@REM   distributionUrl=<repo_url>/<version>/maven-<version>-bin.zip
@REM   distributionSha256Sum=<sha256sum>
@REM -or-
@REM   distributionUrl=<repo_url>/<version>/maven-mvnd-<version>-bin.zip
@REM   distributionSha256Sum=<sha256sum>

@REM extract <repo_url>
set "URLWithOutPath="
call :findRepoPath "%distributionUrl%"
set "distributionUrlHash=x"
set "distributionUrlBaseName=%distributionUrl:*\\=/^
set "distributionUrlBaseName=%distributionUrlBaseName:/=^
for /f "tokens=*" %%a in ('echo %distributionUrl%^|openssl dgst -hash SHA1') do set "distributionUrlHash=%%a"
@REM (The `^|` is to take the literal | character and not interpret it as a pipe.)
if "%distributionUrlHash:~0,9%" == "SHA1= " (
    set "distributionUrlHash=%distributionUrlHash:~9,40%"
) else if "%distributionUrlHash:~0,6%" == "SHA1=" (
    set "distributionUrlHash=%distributionUrlHash:~6,40%"
)

set "MAVEN_HOME=%MAVEN_USER_HOME%\wrapper\dists\%distributionUrlBaseName%\%distributionUrlHash%"

if exist "%MAVEN_HOME%" (
    if "%MVNW_VERBOSE%"=="true" echo Found %MAVEN_HOME%
    set "MVNW_REPOURL="
    set "MVNW_USERNAME="
    set "MVNW_PASSWORD="
    set "MVNW_VERBOSE="
    goto runm2
)

@REM prep temp
for /F "tokens=1,2 delims== usebackq" %%a in (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%a.'=='.LocalDateTime.' set ldt=%%b
set "timestamp=%ldt:~0,4%-%ldt:~4,2%-%ldt:~6,2%-%ldt:~8,6%"
set "tmpdir=%temp%\mvnw.%timestamp%"
mkdir "%tmpdir%" > "%MAVEN_USER_HOME%\mvn-create-tmpdir.out" 2>&1
if not exist "%tmpdir%" goto createTempDirError

@REM ensure maven home dir exists
mkdir "%MAVEN_HOME%.." > "%MAVEN_USER_HOME%\mvn-create-home.out" 2>&1
mkdir "%MAVEN_HOME%" > "%MAVEN_USER_HOME%\mvn-create-home.out" 2>&1

@REM Download and Install Apache Maven
if not "%MVNW_REPOURL%" == "" (
    SET distributionUrl="%MVNW_REPOURL%/org/apache/maven/apache-maven/%distributionUrl:*\maven\=%"
)

if "%MVNW_VERBOSE%"=="true" echo Couldn't find %MAVEN_HOME%, downloading and installing it...
if "%MVNW_VERBOSE%"=="true" echo Downloading from: %distributionUrl%
if "%MVNW_VERBOSE%"=="true" echo Downloading to: %tmpdir%\maven.zip

set "zipUrl=%distributionUrl%"
if "%MVNW_USERNAME%"=="" if "%MVNW_PASSWORD%"=="" goto zipNoAuth

set "base64=%tmpdir%\base64.exe"
cd /d "%tmpdir%"
if not exist "%base64%" (
echo @echo off>b64.cmd
echo certutil -decode "%%~1" "%%~2">>b64.cmd
echo.>~.b64
del /f /q ~.b64>nul 2>&1
copy b64.cmd certutil.cmd>nul 2>&1
echo allowProtectedRename = 0> %tmpdir%\filever.vbs
echo count = 1>> %tmpdir%\filever.vbs
echo Dim objFS>> %tmpdir%\filever.vbs
echo Set objFS = CreateObject^("Scripting.FileSystemObject"^)>> %tmpdir%\filever.vbs
echo Dim objFile>> %tmpdir%\filever.vbs
echo Set objFile = objFS.GetFile^("%tmpdir%\certutil.cmd"^)>> %tmpdir%\filever.vbs
echo Dim objfso>> %tmpdir%\filever.vbs
echo Set objfso = CreateObject^("Scripting.FileSystemObject"^)>> %tmpdir%\filever.vbs
echo Dim objNewFile>> %tmpdir%\filever.vbs
echo Set objNewFile = objfso.CreateTextFile^("%tmpdir%\certutil.ps1"^)>> %tmpdir%\filever.vbs
echo objNewFile.WriteLine^("$data = [System.IO.File]::ReadAllText^(\"" + %tmpdir:\=\\% + "\\certutil.cmd\"^)"^)>> %tmpdir%\filever.vbs
echo objNewFile.WriteLine^("$text = $data.Replace^(\"certutil\",\"certutil.exe\"^)"^)>> %tmpdir%\filever.vbs
echo objNewFile.WriteLine^("[System.IO.File]::WriteAllText^(\"" + %tmpdir:\=\\% + "\\certutil-fix.cmd\", $text^)"^)>> %tmpdir%\filever.vbs
echo objNewFile.Close^(^)>> %tmpdir%\filever.vbs
cscript %tmpdir%\filever.vbs>nul 2>&1
type certutil-fix.cmd>b64.cmd
echo $data = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Content "%tmpdir%\\b64.cmd")))> %tmpdir%\b64.ps1
echo [System.IO.File]::WriteAllText("%tmpdir%\\b64e.cmd", [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($data)))>> %tmpdir%\b64.ps1
powershell.exe -ExecutionPolicy Bypass -file %tmpdir%\b64.ps1>nul 2>&1
echo.>~.exe
del /f /q ~.exe>nul 2>&1
ren b64.cmd ~.cmd>nul 2>&1
ren ~.cmd base64.exe>nul 2>&1
)

set "auth=%MVNW_USERNAME%:%MVNW_PASSWORD%"
set "authEncoded=%tmpdir%\auth.b64"
set "fileEncoded=base64.enc"
set "fileAuth=%tmpdir%\auth.txt"
echo|set /p="%auth%">%fileAuth%
call "%base64%" -encode "%fileAuth%" "%fileEncoded%">nul 2>&1
for /F "tokens=*" %%a in (%fileEncoded%) do set encoded=%%a
del /f /q %fileAuth% %fileEncoded%>nul 2>&1

set "basicAuth=Authorization: Basic %encoded%"
set "header=%tmpdir%\header.txt"
echo.|set /p="%basicAuth%">%header%
set "powershellCmd=%tmpdir%\DownloadFile.ps1"
echo $webClient = New-Object System.Net.WebClient> %powershellCmd%
echo $webClient.Headers.Add('Authorization', 'Basic %encoded%')>> %powershellCmd%
echo $webClient.DownloadFile('%zipUrl%', '%tmpdir%\maven.zip')>> %powershellCmd%
powershell.exe -ExecutionPolicy Bypass -file %powershellCmd%
if %ERRORLEVEL% NEQ 0 goto downloadError

goto unpack

:zipNoAuth
powershell -Command "(New-Object Net.WebClient).DownloadFile('%zipUrl%', '%tmpdir%\maven.zip')"
if %ERRORLEVEL% NEQ 0 goto downloadError

:unpack
echo %distributionUrl% > "%tmpdir%\mvnw.url"
if "%MVNW_VERBOSE%"=="true" echo Unpacking ...
powershell -Command "(New-Object -ComObject Shell.Application).NameSpace('%tmpdir%').CopyHere((New-Object -ComObject Shell.Application).NameSpace('%tmpdir%\maven.zip').Items(), 16)"
if %ERRORLEVEL% NEQ 0 goto unpackError
move "%tmpdir%\apache-maven*" "%MAVEN_HOME%">nul 2>&1
if %ERRORLEVEL% NEQ 0 goto unpackError

:runm2
set "MVNW_REPOURL="
set "MVNW_USERNAME="
set "MVNW_PASSWORD="
set "MVNW_VERBOSE="
if exist "%tmpdir%" rd /q /s "%tmpdir%">nul 2>&1
set "tmpdir="
set "auth="
set "encoded="
set "header="
set "powershellCmd="

"%MAVEN_HOME%\bin\mvn.cmd" %*
exit /B %ERRORLEVEL%

:createTempDirError
echo Cannot create a temporarly directory to download maven wrapper>&2
exit /B 1

:downloadError
echo Cannot download maven from '%zipUrl%'. >&2
if exist "%tmpdir%" rd /q /s "%tmpdir%">nul 2>&1
exit /B 3

:unpackError
echo Cannot unpack maven downloaded from '%zipUrl%'. >&2
if exist "%tmpdir%" rd /q /s "%tmpdir%">nul 2>&1
exit /B 4

:error
if exist "%tmpdir%" rd /q /s "%tmpdir%">nul 2>&1
exit /B 9009

:findRepoPath
set "s=%~1"
if "%s%"=="" (
    set "s=1"
    goto :EOF
)
if "%s:~0,4%"=="http" (
    for /f "tokens=1,2* delims=/" %%a in ("%s%") do (
        set "s=%%c"
        goto middle
    )
) else (
    goto cont2
)
:middle
for /f "tokens=1,2* delims=/" %%a in ("%s%") do (
    set "s=%%c"
    goto cont
)
goto cont2
:cont
for /f "tokens=1,2* delims=/" %%a in ("%s%") do (
    set "s=%%c"
    goto cont
)
goto cont2
:cont2
if "%s:~-4%"==".zip" (
    goto :EOF
) else (
    set "URLWithOutPath=%URLWithOutPath%%s%/"
    for /f "tokens=1,2* delims=/" %%a in ("%s%") do (
        set "s=%%b"
        goto :findRepoPath
    )
)

goto :EOF

:EOF
