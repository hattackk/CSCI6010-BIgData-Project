# Get the directory where the script resides
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# File containing repository URLs (one per line)
$urlFile = "model_repo_urls.txt"

# Check if the file exists
if (-not (Test-Path $urlFile -PathType Leaf)) {
    Write-Host "File '$urlFile' not found. Exiting..."
    exit 1
}

# Read each line from the file and clone the repository
foreach ($url in Get-Content $urlFile) {
    Write-Host "Cloning repository from: $url"
    cd scriptDir
    git clone $url
    # Check if the clone was successful
    if ($LastExitCode -eq 0) {
        Write-Host "Repository cloned successfully."
    } else {
        Write-Host "Failed to clone repository."
    }
}