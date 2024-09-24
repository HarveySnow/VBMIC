$encpath = "D:\code\compress\bpg\bpg-0.9.8-win64\bpgenc.exe"
$decpath = "D:\code\compress\bpg\bpg-0.9.8-win64\bpgdec.exe"

$targetDirectory = "D:\code\compress\bpg\output-city"
$outputDirectory = "D:\code\compress\bpg\decoded"

# 确保输出目录存在
if (-not (Test-Path -Path $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

# 获取目标目录及其子目录下的所有 .bpg 文件
$files = Get-ChildItem -Path $targetDirectory -Recurse -File -Filter "*.bpg"

# 遍历文件列表，进行解码
foreach ($file in $files) {
    # 获取文件的相对路径
    $relativePath = $file.FullName.Substring($targetDirectory.Length + 1)
    
    # 获取仅文件名（无扩展名）
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($relativePath)
    
    # 构建输出文件的完整路径
    $outputFile = Join-Path -Path $outputDirectory -ChildPath "$relativePath.png"
    
    # 确保输出目录存在
    $outputDir = [System.IO.Path]::GetDirectoryName($outputFile)
    if (-not (Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }

    # 调用 bpgdec.exe 进行解码
    & $decpath -o $outputFile $file.FullName

    # 输出解码后的文件路径
    Write-Output "Decoded: $outputFile"
}
