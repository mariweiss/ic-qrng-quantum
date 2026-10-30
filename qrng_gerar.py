# ============================================================
# QRNG + Transformer — IC Computação Quântica
# Inatel — Mariana Bissaro Weiss
# ============================================================

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
import os
from datetime import datetime

# Pasta com timestamp para cada execução
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"resultados/{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Pasta criada: {output_dir}")

# --- Conexão IBM Quantum ---
IBM_TOKEN = os.getenv("IBM_TOKEN")
QiskitRuntimeService.save_account(
    channel='ibm_cloud',
    token=IBM_TOKEN,
    overwrite=True
)

service = QiskitRuntimeService(channel='ibm_cloud')
backend = service.least_busy(operational=True, simulator=False)
print(f"✓ Conectado: {backend.name}")

# --- Circuito Quântico (Hadamard + medição) ---
NUM_QUBITS = 8
qc = QuantumCircuit(NUM_QUBITS)
qc.h(range(NUM_QUBITS))
qc.measure_all()

# Transpila para o backend -> converte para o formato que o hardware consiga executar
print("Transpilando circuito...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
qc_isa = pm.run(qc) # circuito traduzido para o hardware
print("✓ Circuito transpilado")

# --- Geração dos bits (16 jobs x 8192 shots x 8 qubits = ~1M bits) --- -> precisa de 1M bits para treinar e testar com NIST STS
sampler = SamplerV2(backend)
NUM_JOBS = 16
SHOTS = 8192
all_bits = []

for i in range(NUM_JOBS):
    print(f"Job {i+1}/{NUM_JOBS}...")
    job = sampler.run([qc_isa], shots=SHOTS)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    for bitstring, count in counts.items():
        bits_list = [int(b) for b in bitstring]
        all_bits.extend(bits_list * count)

bits = np.array(all_bits)
np.random.shuffle(bits)
np.save(f'{output_dir}/bits_qrng.npy', bits)

print(f"✓ {len(bits):,} bits gerados e salvos em {output_dir}/bits_qrng.npy")
print(f"  Zeros: {(bits==0).sum()/len(bits)*100:.2f}%")
print(f"  Uns:   {(bits==1).sum()/len(bits)*100:.2f}%")