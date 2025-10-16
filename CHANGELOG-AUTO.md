# Changelog - Sistema de Instalação Automática

## Versão 0.1.4 (Nova)

### 🎉 Nova Funcionalidade: Instalação Automática em C:\

#### Mudanças Principais

**1. Download e Instalação Automáticos**
- As bibliotecas agora são instaladas em locais permanentes do sistema:
  - Windows: `C:\libtorch` e `C:\libtensorflow`
  - Linux: `/opt/libtorch` e `/opt/libtensorflow`
- Não há mais subpastas duplicadas com o mesmo nome
- Downloads são feitos em pasta temporária e depois movidos

**2. Configuração Automática de Variáveis de Ambiente**
- `LIBTORCH` → Caminho para LibTorch
- `TENSORFLOW_ROOT` → Caminho para TensorFlow
- `PATH` → Automaticamente atualizado com os diretórios `lib` e `bin`
- Variáveis persistem entre sessões do terminal

**3. Novos Scripts e Documentação**

Scripts criados:
- `setup-env.ps1` → Configura variáveis de ambiente manualmente
- `test-auto-install.ps1` → Verifica se tudo foi instalado corretamente

Documentação criada:
- `AUTO-INSTALL.md` → Guia completo do sistema de instalação automática

Documentação atualizada:
- `README.md` → Seção de instalação atualizada
- `INSTALLATION.md` → Instruções detalhadas do novo sistema

#### Melhorias no build.rs

**Funções Adicionadas:**
- `move_directory()` → Move diretórios inteiros
- `copy_directory_recursive()` → Copia diretórios recursivamente
- `set_environment_variables()` → Configura variáveis de ambiente (Windows)
- `add_to_path_windows()` → Adiciona diretórios ao PATH (Windows)

**Lógica Atualizada:**
- Verifica se as bibliotecas já existem antes de baixar
- Extrai em diretório temporário e move para local permanente
- Remove subpastas duplicadas automaticamente
- Limpa arquivos temporários após instalação
- Configura variáveis de ambiente automaticamente
- Atualiza PATH do usuário com os diretórios necessários

#### Benefícios

✅ **Instalação única**: Bibliotecas ficam em local permanente, não precisam ser baixadas novamente
✅ **Sem configuração manual**: Variáveis de ambiente configuradas automaticamente
✅ **Organização limpa**: Sem subpastas duplicadas (apenas `C:\libtorch`, não `C:\libtorch\libtorch`)
✅ **PATH atualizado**: DLLs encontradas automaticamente pelo sistema
✅ **Persistência**: Configurações permanecem entre sessões

#### Compatibilidade

- ✅ Windows 10/11 (PowerShell 5.1+)
- ✅ Linux (Ubuntu, Fedora, Arch, etc.)
- ⚠️ Pode requerer permissões de administrador/sudo

#### Notas de Migração

Se você já tem uma versão anterior instalada:

1. **Opcional**: Delete as pastas antigas em `deps/`
2. Delete as pastas antigas em `C:\` se já existirem
3. Execute: `cargo clean && cargo build --features libtorch,tensorflow`
4. Execute: `.\setup-env.ps1` (Windows) para verificar variáveis
5. Reinicie o terminal/IDE

#### Arquivos Modificados

```
build.rs              → Lógica de instalação automática
README.md             → Seção de instalação atualizada
INSTALLATION.md       → Instruções detalhadas
AUTO-INSTALL.md       → Novo documento (guia completo)
setup-env.ps1         → Novo script (configuração manual)
test-auto-install.ps1 → Novo script (teste de instalação)
CHANGELOG-AUTO.md     → Este arquivo
```

#### Próximos Passos

- [ ] Adicionar suporte para macOS
- [ ] Cache de downloads para CI/CD
- [ ] Verificação de checksums dos arquivos baixados
- [ ] Suporte para versões customizadas das bibliotecas
- [ ] Interface de linha de comando para gerenciar instalações

---

## Como Usar

### Primeira Instalação

```bash
# Clone o repositório
git clone https://github.com/CopperRS/AI-Copper.git
cd AI-Copper

# Build (instala automaticamente)
cargo build --features libtorch,tensorflow

# Teste a instalação
.\test-auto-install.ps1  # Windows
```

### Verificar Instalação

```powershell
# Windows
$env:LIBTORCH
$env:TENSORFLOW_ROOT
Test-Path C:\libtorch
Test-Path C:\libtensorflow

# Executar teste completo
.\test-auto-install.ps1
```

### Configurar Manualmente (Se Necessário)

```powershell
# Windows
.\setup-env.ps1

# Reinicie o terminal após executar
```

---

## Suporte

Para problemas ou dúvidas:
- Consulte [AUTO-INSTALL.md](AUTO-INSTALL.md)
- Consulte [INSTALLATION.md](INSTALLATION.md)
- Abra uma issue no GitHub

---

**Data**: Outubro 2025  
**Autor**: CopperRS  
**Versão**: 0.1.4
