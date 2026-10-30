import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# --- Dados dos resultados ---
pasta = sorted(os.listdir('resultados'))[-1]

# Min-Entropia
bits = np.load(f'resultados/{pasta}/bits_qrng.npy')
p_max = max(bits.mean(), 1 - bits.mean())
h_min = -np.log2(p_max)

# Acurácia Transformer (resultado obtido)
acc_transformer = 50.03

# NIST
nist_passaram = 8
nist_total = 15

# --- Gráfico ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Resumo da Qualidade do QRNG — IBM Quantum', fontsize=14, fontweight='bold')

# 1. Min-Entropia
ax1 = axes[0]
ax1.bar(['Min-Entropia'], [h_min], color='steelblue', width=0.4)
ax1.axhline(1.0, color='red', linestyle='--', label='Ideal (1.0)')
ax1.set_ylim(0.99, 1.001)
ax1.set_title('Min-Entropia')
ax1.set_ylabel('bits/bit')
ax1.legend()
ax1.text(0, h_min + 0.0001, f'{h_min:.4f}', ha='center', fontsize=11)

# 2. Transformer
ax2 = axes[1]
ax2.bar(['Transformer'], [acc_transformer], color='tomato', width=0.4)
ax2.axhline(50.0, color='red', linestyle='--', label='Ideal (50%)')
ax2.set_ylim(45, 55)
ax2.set_title('Acurácia do Transformer')
ax2.set_ylabel('Acurácia (%)')
ax2.legend()
ax2.text(0, acc_transformer + 0.1, f'{acc_transformer:.2f}%', ha='center', fontsize=11)

# 3. NIST STS
ax3 = axes[2]
ax3.bar(['Passaram', 'Falharam'], [nist_passaram, nist_total - nist_passaram],
        color=['steelblue', 'tomato'])
ax3.set_title('Testes NIST STS')
ax3.set_ylabel('Número de testes')
ax3.set_ylim(0, 16)
for i, v in enumerate([nist_passaram, nist_total - nist_passaram]):
    ax3.text(i, v + 0.2, str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'resultados/{pasta}/grafico_comparativo_final.png', dpi=150)
plt.show()
print(f"✓ Gráfico salvo em resultados/{pasta}/grafico_comparativo_final.png")