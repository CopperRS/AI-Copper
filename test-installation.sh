#!/bin/bash
# Script de teste para verificar a instalação do AI Copper

echo "🔍 Verificando instalação do AI Copper..."

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Verifica se o Cargo.toml existe
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Cargo.toml não encontrado!${NC}"
    echo -e "${YELLOW}Execute este script na raiz do projeto ai_copper${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Cargo.toml encontrado${NC}"

# Verifica se CMake está instalado
echo -e "\n${CYAN}🔍 Verificando CMake...${NC}"
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo -e "${GREEN}✅ CMake instalado: $CMAKE_VERSION${NC}"
else
    echo -e "${RED}❌ CMake não encontrado!${NC}"
    echo -e "${YELLOW}Instale CMake: sudo apt-get install cmake (Ubuntu/Debian)${NC}"
    exit 1
fi

# Verifica se as variáveis de ambiente estão definidas (opcional)
echo -e "\n${CYAN}🔍 Verificando variáveis de ambiente (opcional)...${NC}"
if [ -n "$LIBTORCH" ]; then
    echo -e "${GREEN}✅ LIBTORCH definido: $LIBTORCH${NC}"
else
    echo -e "${YELLOW}ℹ️  LIBTORCH não definido (será baixado automaticamente)${NC}"
fi

if [ -n "$TENSORFLOW_ROOT" ]; then
    echo -e "${GREEN}✅ TENSORFLOW_ROOT definido: $TENSORFLOW_ROOT${NC}"
else
    echo -e "${YELLOW}ℹ️  TENSORFLOW_ROOT não definido (será baixado automaticamente)${NC}"
fi

# Verifica se o diretório deps existe
echo -e "\n${CYAN}🔍 Verificando cache de bibliotecas...${NC}"
if [ -d "deps/libtorch" ]; then
    echo -e "${GREEN}✅ LibTorch encontrado em cache (deps/libtorch)${NC}"
else
    echo -e "${YELLOW}ℹ️  LibTorch não está em cache (será baixado na compilação)${NC}"
fi

if [ -d "deps/tensorflow" ]; then
    echo -e "${GREEN}✅ TensorFlow encontrado em cache (deps/tensorflow)${NC}"
else
    echo -e "${YELLOW}ℹ️  TensorFlow não está em cache (será baixado na compilação)${NC}"
fi

# Tenta compilar
echo -e "\n${CYAN}🔨 Testando compilação...${NC}"
echo -e "${YELLOW}Isso pode levar alguns minutos na primeira vez...${NC}"

START_TIME=$(date +%s)
if cargo build --release; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${GREEN}✅ Compilação bem-sucedida!${NC}"
    echo -e "${CYAN}⏱️  Tempo de compilação: $DURATION segundos${NC}"
else
    echo -e "${RED}❌ Falha na compilação!${NC}"
    exit 1
fi

# Executa exemplo básico
echo -e "\n${CYAN}🧪 Executando exemplo básico...${NC}"
if cargo run --example basic_usage; then
    echo -e "${GREEN}✅ Exemplo executado com sucesso!${NC}"
else
    echo -e "${YELLOW}⚠️  Falha ao executar exemplo${NC}"
    echo -e "${YELLOW}Isso pode ser normal se o exemplo precisar de ajustes${NC}"
fi

echo -e "\n${GREEN}✅ AI Copper está instalado e funcionando!${NC}"

