$encpath = "D:\code\compress\bpg\bpg-0.9.8-win64\bpgenc.exe"
$decpath = "D:\code\compress\bpg\bpg-0.9.8-win64\bpgdec.exe"

$targetDirectory = "D:\code\compress\bpg\output"
$outputDirectory = "D:\code\compress\bpg\decoded"

if (-not (Test-Path -Path $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

$files = Get-ChildItem -Path $targetDirectory -Recurse -File -Filter "*.bpg"

foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($targetDirectory.Length + 1)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($relativePath)
    $outputFile = Join-Path -Path $outputDirectory -ChildPath "$relativePath.png"
    
    $outputDir = [System.IO.Path]::GetDirectoryName($outputFile)
    if (-not (Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }

    & $decpath -o $outputFile $file.FullName
    Write-Output "Decoded: $outputFile"
}
