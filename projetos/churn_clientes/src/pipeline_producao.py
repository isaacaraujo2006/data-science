import os
import subprocess
import time

# Definir o caminho dos scripts
scripts = [
    'C:/Github/data-science/projetos/churn_clientes/src/01_tratamento_limpeza.py',
    'C:/Github/data-science/projetos/churn_clientes/src/02_eda.py',
    'C:/Github/data-science/projetos/churn_clientes/src/04_treinando_melhor_modelo.py'
]

# Iniciar contagem do tempo total de processamento
inicio_tempo_total = time.time()

# Executar cada script em sequência
for script in scripts:
    try:
        print(f"Executando {script}...")
        subprocess.run(['python', script], check=True)
        print(f"{script} executado com sucesso!\n")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script}: {e}\n")
        break

# Calcular o tempo total de processamento
tempo_total_processamento = time.time() - inicio_tempo_total
horas, rem = divmod(tempo_total_processamento, 3600)
minutos, segundos = divmod(rem, 60)
tempo_total_formatado = f"{int(horas):02}:{int(minutos):02}:{int(segundos):02}"

print(f"Tempo total de processamento: {tempo_total_formatado}")
print("Pipeline de produção concluído!")
