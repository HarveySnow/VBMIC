$encpath = "D:\code\compress\bpg\bpg-0.9.8-win64\bpgenc.exe";
$decpath = "D:\code\compress\bpg\bpg-0.9.8-win64\bpgdec.exe";

$targetDirectory = "D:\code\compress\bpg\compresstest"
$outputDirectory = "D:\code\compress\bpg\output"


# 确保输出目录存在
if (-not (Test-Path -Path $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

# 获取目标目录及其子目录下的所有文件
$files = Get-ChildItem -Path $targetDirectory -Recurse -File

# 遍历文件列表，进行压缩
foreach ($file in $files) {
    # 获取文件的相对路径
    $relativePath = $file.FullName.Substring($targetDirectory.Length + 1)
    
    # 获取仅文件名（无扩展名）
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($relativePath)
    
    # 构建输出文件的完整路径
    $outputFile = Join-Path -Path $outputDirectory -ChildPath "$relativePath.bpg"
    
    # 确保输出目录存在
    $outputDir = [System.IO.Path]::GetDirectoryName($outputFile)
    if (-not (Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }

    # 调用 bpgenc.exe 进行压缩
    & $encpath -o $outputFile $file.FullName

    # 输出压缩后的文件路径
    Write-Output "Compressed: $outputFile"
}


