[CmdletBinding()]
param (
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Determine project root and certs directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$certsDir = Join-Path $projectRoot "certs"

$fullchainPath = Join-Path $certsDir "fullchain.pem"
$privkeyPath = Join-Path $certsDir "privkey.pem"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " SETU -- SSL/TLS Development Certificate Generator" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Target directory: $certsDir"

# Ensure certs directory exists
if (-not (Test-Path $certsDir)) {
    Write-Host "[INFO] Creating certs directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $certsDir -Force | Out-Null
}

# Check if certificates already exist
if ((Test-Path $fullchainPath) -and (Test-Path $privkeyPath) -and (-not $Force)) {
    Write-Host "[OK] SSL certificates already exist:" -ForegroundColor Green
    Write-Host "  * Fullchain: $fullchainPath" -ForegroundColor Gray
    Write-Host "  * PrivKey:   $privkeyPath" -ForegroundColor Gray
    Write-Host "Use -Force switch to regenerate." -ForegroundColor Yellow
    exit 0
}

# Locate OpenSSL executable
$openSslPath = $null
$openSslCmd = Get-Command openssl -ErrorAction SilentlyContinue
if ($openSslCmd) {
    $openSslPath = $openSslCmd.Source
} else {
    $knownLocations = @(
        "C:\Program Files\Git\usr\bin\openssl.exe",
        "C:\Program Files\Git\mingw64\bin\openssl.exe",
        "C:\Program Files\OpenSSL-Win64\bin\openssl.exe",
        "C:\OpenSSL-Win64\bin\openssl.exe"
    )
    foreach ($loc in $knownLocations) {
        if (Test-Path $loc) {
            $openSslPath = $loc
            break
        }
    }
}

if (-not $openSslPath) {
    Write-Host "[ERROR] OpenSSL executable was not found on this system." -ForegroundColor Red
    Write-Host "Please install Git for Windows or OpenSSL." -ForegroundColor Red
    Write-Host "Expected path: C:\Program Files\Git\usr\bin\openssl.exe" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Using OpenSSL at: $openSslPath" -ForegroundColor Gray
Write-Host "[INFO] Generating 2048-bit RSA key and self-signed certificate (365 days)..." -ForegroundColor Yellow

# Disable MSYS path conversion for Git OpenSSL to prevent mangling /CN=...
$env:MSYS_NO_PATHCONV = "1"

try {
    $sslArgs = @(
        "req",
        "-x509",
        "-nodes",
        "-days", "365",
        "-newkey", "rsa:2048",
        "-keyout", $privkeyPath,
        "-out", $fullchainPath,
        "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1"
    )

    & $openSslPath $sslArgs

    if ((Test-Path $fullchainPath) -and (Test-Path $privkeyPath)) {
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "[SUCCESS] SSL/TLS certificates generated successfully!" -ForegroundColor Green
        Write-Host "  * Certificate: $fullchainPath" -ForegroundColor White
        Write-Host "  * Private Key: $privkeyPath" -ForegroundColor White
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "Ready for Docker Compose HTTPS mounting: ./certs:/etc/nginx/certs:ro" -ForegroundColor Cyan
    } else {
        throw "Certificate files were not created."
    }
} catch {
    Write-Host "[ERROR] Certificate generation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
