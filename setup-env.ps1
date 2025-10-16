# Script para configurar variáveis de ambiente para AI-Copper
# Execute como Administrador se necessário

Write-Host "===== Configuração de Ambiente AI-Copper =====" -ForegroundColor Cyan
Write-Host ""

# Verifica se está executando como administrador
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if ($isAdmin) {
    Write-Host "Executando como Administrador" -ForegroundColor Green
    $scope = "Machine"
} else {
    Write-Host "Executando como Usuário (recomendado executar como Admin para PATH do sistema)" -ForegroundColor Yellow
    $scope = "User"
}

Write-Host ""

# Configuração LibTorch
$libtorchPath = "C:\libtorch"
if (Test-Path $libtorchPath) {
    Write-Host "LibTorch encontrado em: $libtorchPath" -ForegroundColor Green
    
    # Diretórios
    $libPath = Join-Path $libtorchPath "lib"
    $binPath = Join-Path $libtorchPath "bin"
    $includePath = Join-Path $libtorchPath "include"
    
    # 1. TORCH_HOME - Raiz da instalação
    [Environment]::SetEnvironmentVariable("TORCH_HOME", $libtorchPath, $scope)
    Write-Host "  ✓ TORCH_HOME definida" -ForegroundColor Green
    
    # 2. LD_LIBRARY_PATH - Caminho das libs (.so) no Linux/macOS
    [Environment]::SetEnvironmentVariable("LD_LIBRARY_PATH", $libPath, $scope)
    Write-Host "  ✓ LD_LIBRARY_PATH definida" -ForegroundColor Green
    
    # 3. LIBRARY_PATH - Caminho de linkagem
    [Environment]::SetEnvironmentVariable("LIBRARY_PATH", $libPath, $scope)
    Write-Host "  ✓ LIBRARY_PATH definida" -ForegroundColor Green
    
    # 4. CPATH - Caminho dos headers
    if (Test-Path $includePath) {
        [Environment]::SetEnvironmentVariable("CPATH", $includePath, $scope)
        Write-Host "  ✓ CPATH definida" -ForegroundColor Green
    }
    
    # 5. CMAKE_PREFIX_PATH - Para CMake
    [Environment]::SetEnvironmentVariable("CMAKE_PREFIX_PATH", $libtorchPath, $scope)
    Write-Host "  ✓ CMAKE_PREFIX_PATH definida" -ForegroundColor Green
    
    # 6. PATH - Adiciona lib e bin
    $currentPath = [Environment]::GetEnvironmentVariable("Path", $scope)
    $pathsToAdd = @()
    
    if ((Test-Path $libPath) -and ($currentPath -notlike "*$libPath*")) {
        $pathsToAdd += $libPath
    }
    
    if ((Test-Path $binPath) -and ($currentPath -notlike "*$binPath*")) {
        $pathsToAdd += $binPath
    }
    
    if ($pathsToAdd.Count -gt 0) {
        $newPath = $currentPath + ";" + ($pathsToAdd -join ";")
        [Environment]::SetEnvironmentVariable("Path", $newPath, $scope)
        Write-Host "  ✓ PATH atualizado: $($pathsToAdd -join ', ')" -ForegroundColor Green
    } else {
        Write-Host "  ✓ PATH já configurado" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "  Variáveis configuradas:" -ForegroundColor Cyan
    Write-Host "    TORCH_HOME = $libtorchPath" -ForegroundColor Gray
    Write-Host "    LD_LIBRARY_PATH = $libPath" -ForegroundColor Gray
    Write-Host "    LIBRARY_PATH = $libPath" -ForegroundColor Gray
    if (Test-Path $includePath) {
        Write-Host "    CPATH = $includePath" -ForegroundColor Gray
    }
    Write-Host "    CMAKE_PREFIX_PATH = $libtorchPath" -ForegroundColor Gray
    Write-Host "    PATH += lib, bin" -ForegroundColor Gray
} else {
    Write-Host "LibTorch não encontrado em: $libtorchPath" -ForegroundColor Red
    Write-Host "  Execute o build do projeto para baixar automaticamente" -ForegroundColor Yellow
}

Write-Host ""

# Configuração TensorFlow
$tensorflowPath = "C:\libtensorflow"
if (Test-Path $tensorflowPath) {
    Write-Host "TensorFlow encontrado em: $tensorflowPath" -ForegroundColor Green
    
    # Diretórios
    $libPath = Join-Path $tensorflowPath "lib"
    $binPath = Join-Path $tensorflowPath "bin"
    $includePath = Join-Path $tensorflowPath "include"
    
    # 1. TENSORFLOW_ROOT - Raiz da instalação
    [Environment]::SetEnvironmentVariable("TENSORFLOW_ROOT", $tensorflowPath, $scope)
    Write-Host "  ✓ TENSORFLOW_ROOT definida" -ForegroundColor Green
    
    # 2. LD_LIBRARY_PATH - Caminho das libs (.so)
    [Environment]::SetEnvironmentVariable("LD_LIBRARY_PATH", $libPath, $scope)
    Write-Host "  ✓ LD_LIBRARY_PATH definida" -ForegroundColor Green
    
    # 3. LIBRARY_PATH - Caminho de linkagem
    [Environment]::SetEnvironmentVariable("LIBRARY_PATH", $libPath, $scope)
    Write-Host "  ✓ LIBRARY_PATH definida" -ForegroundColor Green
    
    # 4. CPATH - Caminho dos headers
    if (Test-Path $includePath) {
        [Environment]::SetEnvironmentVariable("CPATH", $includePath, $scope)
        Write-Host "  ✓ CPATH definida" -ForegroundColor Green
    }
    
    # 5. PATH - Adiciona lib e bin
    $currentPath = [Environment]::GetEnvironmentVariable("Path", $scope)
    $pathsToAdd = @()
    
    if ((Test-Path $libPath) -and ($currentPath -notlike "*$libPath*")) {
        $pathsToAdd += $libPath
    }
    
    if ((Test-Path $binPath) -and ($currentPath -notlike "*$binPath*")) {
        $pathsToAdd += $binPath
    }
    
    if ($pathsToAdd.Count -gt 0) {
        $newPath = $currentPath + ";" + ($pathsToAdd -join ";")
        [Environment]::SetEnvironmentVariable("Path", $newPath, $scope)
        Write-Host "  ✓ PATH atualizado: $($pathsToAdd -join ', ')" -ForegroundColor Green
    } else {
        Write-Host "  ✓ PATH já configurado" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "  Variáveis configuradas:" -ForegroundColor Cyan
    Write-Host "    TENSORFLOW_ROOT = $tensorflowPath" -ForegroundColor Gray
    Write-Host "    LD_LIBRARY_PATH = $libPath" -ForegroundColor Gray
    Write-Host "    LIBRARY_PATH = $libPath" -ForegroundColor Gray
    if (Test-Path $includePath) {
        Write-Host "    CPATH = $includePath" -ForegroundColor Gray
    }
    Write-Host "    PATH += lib, bin" -ForegroundColor Gray
} else {
    Write-Host "TensorFlow não encontrado em: $tensorflowPath" -ForegroundColor Red
    Write-Host "  Execute o build do projeto para baixar automaticamente" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===== Configuração Concluída =====" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANTE: Reinicie o terminal ou IDE para que as mudanças tenham efeito!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para verificar as variáveis:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  LibTorch:" -ForegroundColor Yellow
Write-Host "    `$env:TORCH_HOME" -ForegroundColor White
Write-Host "    `$env:LD_LIBRARY_PATH" -ForegroundColor White
Write-Host "    `$env:LIBRARY_PATH" -ForegroundColor White
Write-Host "    `$env:CPATH" -ForegroundColor White
Write-Host "    `$env:CMAKE_PREFIX_PATH" -ForegroundColor White
Write-Host ""
Write-Host "  TensorFlow:" -ForegroundColor Yellow
Write-Host "    `$env:TENSORFLOW_ROOT" -ForegroundColor White
Write-Host "    `$env:LD_LIBRARY_PATH" -ForegroundColor White
Write-Host "    `$env:LIBRARY_PATH" -ForegroundColor White
Write-Host "    `$env:CPATH" -ForegroundColor White
Write-Host ""
Write-Host "  Sistema:" -ForegroundColor Yellow
Write-Host "    `$env:PATH" -ForegroundColor White
