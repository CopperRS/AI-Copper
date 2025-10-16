#!/bin/bash
# Script para configurar variáveis de ambiente para AI-Copper no macOS/Linux
# Execute: source ./setup-env.sh

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
GRAY='\033[0;37m'
NC='\033[0m'

echo -e "${CYAN}===== Configuração de Ambiente AI-Copper (macOS/Linux) =====${NC}\n"

# URLs oficiais (ajuste as versões conforme necessário)
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.1.zip"
TENSORFLOW_URL="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.11.0.tar.gz"

LIBTORCH_PATH="/usr/local/libtorch"
TENSORFLOW_PATH="/usr/local/libtensorflow"

# Função para baixar e extrair LibTorch
download_libtorch() {
    if [ ! -d "$LIBTORCH_PATH" ]; then
        echo -e "${CYAN}Baixando LibTorch...${NC}"
        curl -L "$LIBTORCH_URL" -o /tmp/libtorch.zip
        sudo unzip -q /tmp/libtorch.zip -d /usr/local/
        sudo mv /usr/local/libtorch /usr/local/libtorch_tmp
        sudo mv /usr/local/libtorch_tmp/libtorch /usr/local/libtorch
        sudo rm -rf /usr/local/libtorch_tmp
        rm /tmp/libtorch.zip
        echo -e "${GREEN}LibTorch instalado em $LIBTORCH_PATH${NC}"
    else
        echo -e "${GREEN}LibTorch já está instalado em $LIBTORCH_PATH${NC}"
    fi
}

# Função para baixar e extrair TensorFlow
download_tensorflow() {
    if [ ! -d "$TENSORFLOW_PATH" ]; then
        echo -e "${CYAN}Baixando TensorFlow...${NC}"
        curl -L "$TENSORFLOW_URL" -o /tmp/libtensorflow.tar.gz
        sudo mkdir -p "$TENSORFLOW_PATH"
        sudo tar -C "$TENSORFLOW_PATH" -xzf /tmp/libtensorflow.tar.gz
        rm /tmp/libtensorflow.tar.gz
        echo -e "${GREEN}TensorFlow instalado em $TENSORFLOW_PATH${NC}"
    else
        echo -e "${GREEN}TensorFlow já está instalado em $TENSORFLOW_PATH${NC}"
    fi
}

echo -e "${CYAN}Verificando dependências...${NC}"
download_libtorch
download_tensorflow

echo -e "${CYAN}Configurando variáveis de ambiente...${NC}"
# Configuração de variáveis de ambiente
export TORCH_HOME="$LIBTORCH_PATH"
export LD_LIBRARY_PATH="$LIBTORCH_PATH/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="$LIBTORCH_PATH/lib:${LIBRARY_PATH}"
export CPATH="$LIBTORCH_PATH/include:${CPATH}"
export CMAKE_PREFIX_PATH="$LIBTORCH_PATH"
export PATH="$LIBTORCH_PATH/lib:$LIBTORCH_PATH/bin:$PATH"

export TENSORFLOW_ROOT="$TENSORFLOW_PATH"
export LD_LIBRARY_PATH="$TENSORFLOW_PATH/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="$TENSORFLOW_PATH/lib:${LIBRARY_PATH}"
export CPATH="$TENSORFLOW_PATH/include:${CPATH}"
export PATH="$TENSORFLOW_PATH/lib:$TENSORFLOW_PATH/bin:$PATH"

echo -e "  \u2713 Variáveis de ambiente configuradas"

echo -e "\n${CYAN}===== Configuração Concluída =====${NC}"
echo -e "${YELLOW}IMPORTANTE: Rode 'source ./setup-env.sh' para aplicar as variáveis neste terminal!${NC}"
echo -e "\nPara verificar as variáveis, use: echo $TORCH_HOME, echo $TENSORFLOW_ROOT, echo $LD_LIBRARY_PATH, etc.\n"
