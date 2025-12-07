#!/bin/bash

# Base do projeto
BASE_DIR="emb_nrs"

# Lista de pastas (array)
folders=(
  "data/raw"
  "data/processed"
  "data/embeddings"
  "notebooks"
  "src"
  "models/mistral"
  "env"
  "tests"
  "scripts"
  "logs"
  "docs"
)

# Criar pastas somente se não existirem
for folder in "${folders[@]}"; do
  path="$BASE_DIR/$folder"
  if [ ! -d "$path" ]; then
    mkdir -p "$path"
    echo "Created: $path"
  else
    echo "Already exists: $path"
  fi
done

# Criar arquivos básicos (sobrescreve se já existirem)
echo -e "logs/\nmodels/\ndata/" > "$BASE_DIR/.gitignore"

cat <<EOF > "$BASE_DIR/docs/README.md"
# Projeto Embeddings NRs

Documentação inicial.
EOF

cat <<EOF > "$BASE_DIR/env/environment.yml"
name: emb_nrs
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - sentence-transformers
  - faiss-cpu
  - transformers
EOF

cat <<EOF > "$BASE_DIR/env/requirements.txt"
sentence-transformers
faiss-cpu
transformers
EOF
