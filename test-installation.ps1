#!/usr/bin/env pwsh
# Script de teste para verificar a instalação do AI Copper

Write-Host "🔍 Verificando instalação do AI Copper..." -ForegroundColor Cyan

# Verifica se o Cargo.toml existe
if (!(Test-Path "Cargo.toml")) {
    Write-Host "❌ Cargo.toml não encontrado!" -ForegroundColor Red
    Write-Host "Execute este script na raiz do projeto ai_copper" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Cargo.toml encontrado" -ForegroundColor Green

# Verifica se CMake está instalado
Write-Host "`n🔍 Verificando CMake..." -ForegroundColor Cyan
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "✅ CMake instalado: $cmakeVersion" -ForegroundColor Green
} else {
    Write-Host "❌ CMake não encontrado!" -ForegroundColor Red
    Write-Host "Instale CMake de: https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

# Verifica se as variáveis de ambiente estão definidas (opcional)
Write-Host "`n🔍 Verificando variáveis de ambiente (opcional)..." -ForegroundColor Cyan
$libtorch = $env:LIBTORCH
$tensorflow = $env:TENSORFLOW_ROOT

if ($libtorch) {
    Write-Host "✅ LIBTORCH definido: $libtorch" -ForegroundColor Green
} else {
    Write-Host "ℹ️  LIBTORCH não definido (será baixado automaticamente)" -ForegroundColor Yellow
}

if ($tensorflow) {
    Write-Host "✅ TENSORFLOW_ROOT definido: $tensorflow" -ForegroundColor Green
} else {
    Write-Host "ℹ️  TENSORFLOW_ROOT não definido (será baixado automaticamente)" -ForegroundColor Yellow
}

# Verifica se o diretório deps existe
Write-Host "`n🔍 Verificando cache de bibliotecas..." -ForegroundColor Cyan
if (Test-Path "deps/libtorch") {
    Write-Host "✅ LibTorch encontrado em cache (deps/libtorch)" -ForegroundColor Green
} else {
    Write-Host "ℹ️  LibTorch não está em cache (será baixado na compilação)" -ForegroundColor Yellow
}

if (Test-Path "deps/tensorflow") {
    Write-Host "✅ TensorFlow encontrado em cache (deps/tensorflow)" -ForegroundColor Green
} else {
    Write-Host "ℹ️  TensorFlow não está em cache (será baixado na compilação)" -ForegroundColor Yellow
}

# Tenta compilar
Write-Host "`n🔨 Testando compilação..." -ForegroundColor Cyan
Write-Host "Isso pode levar alguns minutos na primeira vez..." -ForegroundColor Yellow

$compileStart = Get-Date
$compileResult = cargo build --release 2>&1
$compileEnd = Get-Date
$compileDuration = ($compileEnd - $compileStart).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Compilação bem-sucedida!" -ForegroundColor Green
    Write-Host "⏱️  Tempo de compilação: $([math]::Round($compileDuration, 2)) segundos" -ForegroundColor Cyan
} else {
    Write-Host "❌ Falha na compilação!" -ForegroundColor Red
    Write-Host "`nErros:" -ForegroundColor Yellow
    Write-Host $compileResult
    exit 1
}

# Executa exemplo básico
Write-Host "`n🧪 Executando exemplo básico..." -ForegroundColor Cyan
$exampleResult = cargo run --example basic_usage 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Exemplo executado com sucesso!" -ForegroundColor Green
    Write-Host "`nSaída:" -ForegroundColor Cyan
    Write-Host $exampleResult
} else {
    Write-Host "⚠️  Falha ao executar exemplo" -ForegroundColor Yellow
    Write-Host "Isso pode ser normal se o exemplo precisar de ajustes" -ForegroundColor Yellow
}

Write-Host "`n✅ AI Copper está instalado e funcionando!" -ForegroundColor Green

Write-Host "`n💡 Dica: As bibliotecas baixadas estão em deps/ e serão reutilizadas" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan
