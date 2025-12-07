import subprocess
import os

SRC_DIR = "src"

def run_script(script_name, is_shell=False):
    script_path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"Erro: {script_path} não encontrado.")
        return
    print(f"\n>>> Rodando {script_name}...\n")

    if is_shell:
        result = subprocess.run(["bash", script_path], capture_output=True, text=True)
    else:
        result = subprocess.run(["python", script_path], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Erros:", result.stderr)

def main():
    # Ordem do pipeline
    run_script("preprocessing.py")
    run_script("embeddings.py")
    run_script("run.sh", is_shell=True)  # aqui rodamos como shell

    print("\n>>> Pipeline completo finalizado.\n")

if __name__ == "__main__":
    main()
