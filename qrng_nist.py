import numpy as np
import os
from nistrng import run_all_battery, SP800_22R1A_BATTERY

# Carrega os bits mais recentes
pasta = sorted(os.listdir('resultados'))[-1]
bits = np.load(f'resultados/{pasta}/bits_qrng.npy')
print(f"✓ Bits carregados: {len(bits):,}")

# NIST STS
print("\n--- Testes NIST STS ---")
sequence = np.where(bits > 0, 1, 0).astype(np.int8)
results = run_all_battery(sequence, SP800_22R1A_BATTERY, False)

aprovados = 0
reprovados = 0
for result, _ in results:
    status = "PASSOU" if result.passed else "FALHOU"
    if result.passed:
        aprovados += 1
    else:
        reprovados += 1
    print(f"  {result.name}: {status}")

print(f"\n✓ Resultado: {aprovados} passaram, {reprovados} falharam")

with open(f'resultados/{pasta}/nist_resultados.txt', 'w') as f:
    for result, _ in results:
        f.write(f"{'PASSOU' if result.passed else 'FALHOU'}: {result.name}\n")
    f.write(f"\nTotal: {aprovados} passaram, {reprovados} falharam\n")