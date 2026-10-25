[CmdletBinding()]
param(
    [string]$Tag,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "AXIOM"),
    [switch]$Force,
    [switch]$NoPathUpdate
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Repo = "NetCore-Technologies/AXIOM-AI"
$TempDir = $null

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ("> " + $Message) -ForegroundColor Cyan
}
function Stop-Install([string]$Message) {
    throw ("AXIOM installer: " + $Message)
}

try {
    if ([string]::IsNullOrWhiteSpace($InstallDir)) {
        Stop-Install "Install directory cannot be empty."
    }
    $InstallDir = [IO.Path]::GetFullPath($InstallDir)

    if (-not $Tag) {
        Write-Step "Looking up the latest published AXIOM release"
        $Releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=30" -Headers @{ "User-Agent" = "AXIOM-Installer" }
        $Release = $Releases | Where-Object { -not $_.draft -and -not $_.prerelease } | Select-Object -First 1
        if (-not $Release) {
            $Release = $Releases | Where-Object { -not $_.draft } | Select-Object -First 1
        }
        if (-not $Release) {
            Stop-Install "No published release was found. Check the GitHub Releases page or pass -Tag explicitly."
        }
        $Tag = [string]$Release.tag_name
    } else {
        if ($Tag -notmatch '^v\d+\.\d+\.\d+([.-][0-9A-Za-z.-]+)?$') {
            Stop-Install "Invalid release tag. Expected a version such as v0.3.0-beta.6."
        }
        $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$Tag" -Headers @{ "User-Agent" = "AXIOM-Installer" }
        if ($Release.draft) {
            Stop-Install "Release $Tag is a draft and cannot be installed."
        }
    }

    $AssetName = "AXIOM-$Tag-windows-x64.exe"
    $DownloadUrl = "https://github.com/$Repo/releases/download/$Tag/$AssetName"
    $ChecksumsUrl = "https://github.com/$Repo/releases/download/$Tag/SHA256SUMS"

    Write-Host "AXIOM Windows Installer" -ForegroundColor White
    Write-Host ("Release : " + $Tag) -ForegroundColor Green
    Write-Host ("Target  : " + $InstallDir)
    Write-Host ("Asset   : " + $AssetName)
    if (-not $Force -and (Test-Path (Join-Path $InstallDir "AXIOM.exe"))) {
        $Answer = Read-Host "AXIOM is already installed. Replace it with $Tag? [y/N]"
        if ($Answer -notmatch '^(y|yes)$') {
            Write-Host "Cancelled; existing installation was not changed."
            return
        }
    }

    $TempDir = Join-Path ([IO.Path]::GetTempPath()) ("axiom-install-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    $DownloadPath = Join-Path $TempDir $AssetName
    $ChecksumPath = Join-Path $TempDir "SHA256SUMS"

    Write-Step "Downloading the release binary and checksum manifest"
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $DownloadPath -Headers @{ "User-Agent" = "AXIOM-Installer" }
    Invoke-WebRequest -Uri $ChecksumsUrl -OutFile $ChecksumPath -Headers @{ "User-Agent" = "AXIOM-Installer" }
    if (-not (Test-Path $DownloadPath) -or (Get-Item $DownloadPath).Length -lt 1) {
        Stop-Install "The release download is empty."
    }

    $ChecksumLine = Get-Content $ChecksumPath | Where-Object {
        $_ -match ('\s\*?' + [regex]::Escape($AssetName) + '$')
    } | Select-Object -First 1
    if (-not $ChecksumLine -or $ChecksumLine -notmatch '^([0-9a-fA-F]{64})\s+\*?') {
        Stop-Install "SHA256SUMS does not contain a valid checksum for $AssetName. Refusing to install an unverified binary."
    }
    $ExpectedHash = $Matches[1].ToLowerInvariant()
    $ActualHash = (Get-FileHash -Path $DownloadPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash) {
        Stop-Install "Checksum mismatch for $AssetName. The existing installation has not been changed."
    }
    Write-Host ("SHA-256 verified: " + $ActualHash) -ForegroundColor Green

    Write-Step "Installing the verified binary"
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $Destination = Join-Path $InstallDir "AXIOM.exe"
    $Staged = Join-Path $InstallDir ("AXIOM.exe.new-" + [guid]::NewGuid().ToString("N"))
    Copy-Item -LiteralPath $DownloadPath -Destination $Staged
    try {
        if (Test-Path $Destination) {
            Remove-Item -LiteralPath $Destination -Force
        }
        Move-Item -LiteralPath $Staged -Destination $Destination
    } finally {
        if (Test-Path $Staged) {
            Remove-Item -LiteralPath $Staged -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not $NoPathUpdate) {
        $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $PathEntries = @($UserPath -split ';' | Where-Object { $_ })
        $PathExists = $false
        foreach ($Entry in $PathEntries) {
            if ($Entry.TrimEnd('\') -ieq $InstallDir.TrimEnd('\')) {
                $PathExists = $true
                break
            }
        }
        if (-not $PathExists) {
            $NewPath = (@($PathEntries) + $InstallDir) -join ';'
            [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
            Write-Host "Added the install directory to your user PATH." -ForegroundColor Green
        } else {
            Write-Host "Install directory is already on your user PATH."
        }
    }

    Write-Step "Checking the installed executable"
    & $Destination --no-banner version
    if ($LASTEXITCODE -ne 0) {
        Stop-Install "The executable was installed but its version check failed. Run it manually to diagnose."
    }
    Write-Host ""
    Write-Host ("AXIOM " + $Tag + " installed successfully at " + $Destination) -ForegroundColor Green
    if ($NoPathUpdate) {
        Write-Host "PATH was not changed. Run the executable by its full path."
    } else {
        Write-Host "Open a new PowerShell window, then run: axiom guide"
    }
    Write-Host "For a pinned beta, download this script and run: .\install.ps1 -Tag v0.3.0-beta.6"
} catch {
    Write-Error $_
    exit 1
} finally {
    if ($TempDir -and (Test-Path $TempDir)) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
