# HHL algorithm in math

# ================================
import numpy as np
from scipy.linalg import expm
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import phase_estimation, UnitaryGate
from qiskit.quantum_info import Statevector
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ================================

# Now we have a problem Ax = b

A = np.array([
    [3, 1],
    [1, 3]
])
b = np.array([1, 0], dtype=float)

# solve in classical computation:
x = np.linalg.solve(A, b)
print(x)

# How to solve it in quantum computation? The answer is HHL algorithm.

# First, we need to know that this is a simple case:
# A is Hermitian matrix, and b is a vector.
# So we can see: A|ui> = lambda_i |ui>

# it procide a group computational basis: {|ui>}, and naturally: b = Σ_i B_i |ui>
# it also provide another imformation:A^(-1)|ui> = 1/lambda_i |ui>

# we should recall that x = A^(-1) b, which means: x = Σ_i 1/lambda_i B_i |ui>
# if we want to know the value of x, we need to know lambda_i and B_i(since we have b, B_i is known).
# additionally, here we define |b> = b / ||b|| = Σ_i B_i |ui> / ||b||

# lambda_i remind us of QPE method.
# however, lambda_i is the eigenvalue of A which is not a unitary matrix.

# so we can use a special transformation: U = e^(i At) to make a unitary matrix:
t = np.pi / 4  # the value of t is complicated, here we just skip the detail
U = expm(1j * A * t)
print(U)
# since we have A|ui> = lambda_i |ui>, we can get:
# U|ui> = e^(i lambda_i t) |ui>, and if rewrite it in another way: U|ui> = e^(2*pi*i*phi ) |ui>
# ===> phi = lambda_i t / (2*pi)

# ================================
# now we have U and |b>, with QPE method, we can get: phi = QPE(U, |b>)
num_phase_qubits = 4
phase_register = QuantumRegister(num_phase_qubits)
system_register = QuantumRegister(1)
qc = QuantumCircuit(phase_register, system_register)
qc.initialize(b / np.linalg.norm(b), system_register)
qpe_circuit = qk.circuit.library.phase_estimation(num_phase_qubits, UnitaryGate(U))
qc.compose(qpe_circuit, inplace=True)

# the above procedure is just the QPE method, and we can get:
state_after_qpe = Statevector.from_instruction(qc)
phase_probability = state_after_qpe.probabilities_dict(
    qargs=[qc.find_bit(q).index for q in phase_register]
)
print(phase_probability)
for key, value in phase_probability.items():
    if value > 1e-6:
        print(key, value)

# we can get the result:
# Qiskit output = 0001  phase bits = 1000  phi = 0.5   probability = 0.5
# Qiskit output = 0010  phase bits = 0100  phi = 0.25  probability = 0.5
phi_1 = 0.5
phi_2 = 0.25
lambda_1 = phi_1 * 2*np.pi / t
lambda_2 = phi_2 * 2*np.pi / t
print(lambda_1, lambda_2)

# now reiterate the final goal: x = Σ_i 1/lambda_i B_i |ui>
# so naturally we will think: since we get lambda_i, we can use the value directly.
# but the truth is: the measurement procedure would not be carried out in this way.
# we need to get 1/lambda_i without measurement.

# ================================
# before dealing this problem, we need to summarize the procedure above:
# 1. prepare the state |b>: here we should reiterate the normalization |b> = b / ||b|| = Σ_i B_i |ui> / ||b|| = Σ_i beta_i |ui>
# 2. prepare phase qubits: |b> -> |0000> |b>  -> Σ_i beta_i |0000> |ui>
# 3. apply QPE: Σ_i beta_i |0000>  |ui> -> Σ_i beta_i |phi_i> |ui>
# ================================

# ================================
# now we continue to add a new ancillary qubit: |0> : Σ_i beta_i |phi_i> |ui> |0>
# if we apply rotation to |0>,|0> -> cos(theta/2) |0> + sin(theta/2)) |1>  -> sqrt(1 - C^2/lambda_i^2) |0> + C/lambda_i |1>
# then we can see the term 1/lambda_i appears.
# so the whole state will be: Σ_i beta_i |phi_i> |ui> (sqrt(1 - C^2/lambda_i^2) |0> + C/lambda_i |1>)
# then apply inverse QPE to remove |phi_i>:
# Σ_i beta_i |phi_i> |ui> (...) -> Σ_i beta_i |ui> (...)

# if the ancillary qubit is measured as |1>, the system branch becomes:
# C Σ_i beta_i/lambda_i |ui>

# since A^(-1)|b> = Σ_i beta_i/lambda_i |ui>,
# this branch is proportional to A^(-1)|b>.

# after normalization, we define the remaining system state as |x>:
# |x> = A^(-1)|b> / ||A^(-1)|b>||

# the probability of measuring the ancillary qubit as |1> is:
# P = C^2 ||A^(-1)|b>||^2

# therefore:
# ||A^(-1)|b>|| = sqrt(P) / C

# since |b> = b / ||b||, the original classical solution is:
# x = ||b|| * sqrt(P) / C * |x>