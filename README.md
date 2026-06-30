# Sparse Paged LLM Architecture

Uma implementação de referência em C++20 de uma arquitetura de LLM com **Computação Condicional (MoE)** e **Paginação Hierárquica de Memória (VRAM/RAM/NVMe)**. Este projeto valida a tese de que é possível treinar e inferir modelos maiores que a memória disponível usando roteamento esparso e carregamento sob demanda.

## 🚀 Destaques

- **Arquitetura Esparsa (MoE):** Ativa apenas `Top-K` experts por token, reduzindo FLOPs.
- **Paginação Hierárquica:** Gerencia dados em três níveis (VRAM, RAM, NVMe) com política de substituição baseada em calor (`hot_score`).
- **Treinamento End-to-End:** Implementa backpropagation real através do roteador e experts, provando convergência em tarefas de longo alcance ("Needle in Haystack").
- **Cross-Platform:** Código puro em C++17/20, sem dependências de CUDA ou Linux-specifics, rodando nativamente em Windows e Linux.

## 🏗️ Arquitetura

O sistema é composto por quatro módulos principais:

1.  **`SparseRouter`**: Decide quais experts ativar baseado no estado oculto. Inclui mecanismos de histerese para evitar troca excessiva (thrashing).
2.  **`HierarchicalPager`**: Simula o subsistema de memória. Monitora acessos e promove/evicta blocos entre VRAM, RAM e NVMe.
3.  **`MoELayer`**: Camada de Mistura de Experts que combina saídas ponderadas pelos gates do router.
4.  **`PaginatedModel`**: Orquestra o fluxo de treino, integrando embedding, MoE e paginação de contexto.

## 📊 Resultados do Experimento

Ao executar o teste de validação (`main.cpp`):

- **Convergência:** A perda (Loss) cai drasticamente (>80% de redução) em poucas épocas, provando que o gradiente flui corretamente através do roteamento esparso.
- **Eficiência de Memória:** O Pager demonstra alta taxa de acertos (Hit Rate) na VRAM após aquecimento, minimizando leituras simuladas de NVMe.
- **Sparsity:** Apenas ~25% dos parâmetros são ativos por passo de tempo.

## 🛠️ Como Compilar e Rodar

Requisitos: Compilador C++17 ou superior (GCC, Clang, MSVC).

```bash
# Compile manualmente
g++ -std=c++17 -O3 src/*.cpp -o llm_experiment

# Ou use CMake
mkdir build && cd build
cmake ..
make

# Executar
./llm_experiment
```

## 📈 Próximos Passos (Roadmap)

1.  **Backend de I/O Real:** Substituir a simulação de NVMe por `io_uring` (Linux) ou `IOCP` (Windows).
2.  **Suporte a GPU:** Portar tensores e kernels de atenção para CUDA/Vulkan.
3.  **Atenção HSRA:** Implementar o módulo de Atenção Esparsa Hierárquica conforme especificação.
4.  **Dataset Real:** Treinar em corpora reais (WikiText, Code) em vez de dados sintéticos.

## 📄 Licença

MIT License. Livre para uso acadêmico e comercial.
