# Script de Teste de Instalação Automática
# Verifica se as bibliotecas foram instaladas corretamente

Write-Host "===== Teste de Instalação AI-Copper =====" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Função para verificar diretório
function Test-Directory {
    param($Path, $Name)
    
    Write-Host "Verificando $Name..." -NoNewline
    if (Test-Path $Path) {
        Write-Host " OK" -ForegroundColor Green
        return $true
    } else {
        Write-Host " FALHOU" -ForegroundColor Red
        Write-Host "  Diretório não encontrado: $Path" -ForegroundColor Yellow
        return $false
    }
}

# Função para verificar variável de ambiente
function Test-EnvVar {
    param($VarName, $ExpectedPath)
    
    Write-Host "Verificando variável $VarName..." -NoNewline
    $value = [Environment]::GetEnvironmentVariable($VarName, "User")
    if (-not $value) {
        $value = [Environment]::GetEnvironmentVariable($VarName, "Machine")
    }
    
    if ($value) {
        Write-Host " OK" -ForegroundColor Green
        Write-Host "  Valor: $value" -ForegroundColor Gray
        return $true
    } else {
        Write-Host " FALHOU" -ForegroundColor Red
        Write-Host "  Variável não definida" -ForegroundColor Yellow
        return $false
    }
}

# Função para verificar PATH
function Test-PathEntry {
    param($Path, $Name)
    
    Write-Host "Verificando $Name no PATH..." -NoNewline
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $fullPath = "$userPath;$machinePath"
    
    if ($fullPath -like "*$Path*") {
        Write-Host " OK" -ForegroundColor Green
        return $true
    } else {
        Write-Host " FALHOU" -ForegroundColor Red
        Write-Host "  Caminho não encontrado no PATH" -ForegroundColor Yellow
        return $false
    }
}

# Função para listar DLLs
function List-DLLs {
    param($Path, $Name)
    
    Write-Host ""
    Write-Host "DLLs em $Name ($Path):" -ForegroundColor Cyan
    if (Test-Path $Path) {
        $dlls = Get-ChildItem -Path $Path -Filter "*.dll" -ErrorAction SilentlyContinue
        if ($dlls) {
            $dlls | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor Gray
            }
            Write-Host "  Total: $($dlls.Count) DLLs" -ForegroundColor Green
        } else {
            Write-Host "  Nenhuma DLL encontrada" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Diretório não existe" -ForegroundColor Red
    }
}

Write-Host "=== LibTorch ===" -ForegroundColor Yellow
Write-Host ""
$allGood = (Test-Directory "C:\libtorch" "C:\libtorch") -and $allGood
$allGood = (Test-Directory "C:\libtorch\lib" "C:\libtorch\lib") -and $allGood
$allGood = (Test-Directory "C:\libtorch\include" "C:\libtorch\include") -and $allGood

Write-Host ""
Write-Host "Variáveis de ambiente:" -ForegroundColor Cyan
$allGood = (Test-EnvVar "TORCH_HOME" "C:\libtorch") -and $allGood
$allGood = (Test-EnvVar "LD_LIBRARY_PATH" "C:\libtorch\lib") -and $allGood
$allGood = (Test-EnvVar "LIBRARY_PATH" "C:\libtorch\lib") -and $allGood
$allGood = (Test-EnvVar "CPATH" "C:\libtorch\include") -and $allGood
$allGood = (Test-EnvVar "CMAKE_PREFIX_PATH" "C:\libtorch") -and $allGood

Write-Host ""
Write-Host "PATH entries:" -ForegroundColor Cyan
$allGood = (Test-PathEntry "C:\libtorch\lib" "C:\libtorch\lib") -and $allGood

List-DLLs "C:\libtorch\lib" "LibTorch"

Write-Host ""
Write-Host "=== TensorFlow ===" -ForegroundColor Yellow
Write-Host ""
$allGood = (Test-Directory "C:\libtensorflow" "C:\libtensorflow") -and $allGood
$allGood = (Test-Directory "C:\libtensorflow\lib" "C:\libtensorflow\lib") -and $allGood
$allGood = (Test-Directory "C:\libtensorflow\include" "C:\libtensorflow\include") -and $allGood

Write-Host ""
Write-Host "Variáveis de ambiente:" -ForegroundColor Cyan
$allGood = (Test-EnvVar "TENSORFLOW_ROOT" "C:\libtensorflow") -and $allGood
$allGood = (Test-EnvVar "LD_LIBRARY_PATH" "C:\libtensorflow\lib") -and $allGood
$allGood = (Test-EnvVar "LIBRARY_PATH" "C:\libtensorflow\lib") -and $allGood
$allGood = (Test-EnvVar "CPATH" "C:\libtensorflow\include") -and $allGood

Write-Host ""
Write-Host "PATH entries:" -ForegroundColor Cyan
$allGood = (Test-PathEntry "C:\libtensorflow\lib" "C:\libtensorflow\lib") -and $allGood

List-DLLs "C:\libtensorflow\lib" "TensorFlow"

Write-Host ""
Write-Host "===== Resultado =====" -ForegroundColor Cyan
Write-Host ""

if ($allGood) {
    Write-Host "Todas as verificações passaram! ✓" -ForegroundColor Green
    Write-Host ""
    Write-Host "Você pode compilar o projeto agora:" -ForegroundColor Cyan
    Write-Host "  cargo build --features libtorch,tensorflow" -ForegroundColor White
} else {
    Write-Host "Algumas verificações falharam! ✗" -ForegroundColor Red
    Write-Host ""
    Write-Host "Execute o build para instalar as dependências:" -ForegroundColor Yellow
    Write-Host "  cargo build --features libtorch,tensorflow" -ForegroundColor White
    Write-Host ""
    Write-Host "Ou configure manualmente as variáveis de ambiente:" -ForegroundColor Yellow
    Write-Host "  .\setup-env.ps1" -ForegroundColor White
}

Write-Host ""
