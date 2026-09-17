# 1-order Trotter decomposition

# ==============================
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate, PauliEvolutionGate
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")

# ==============================

# since we have a matrix A which is a hermitian matrix, if we want to do the Hamiltonian simulation,
# we can use the Trotter decomposition to approximate the time evolution operator e^(-iAt)

def Trotter_1st_order(A, t, r):
    # check if A is the hermitian matrix
    if not np.allclose(A, A.conj().T):
        raise ValueError("A is not a hermitian matrix")
    # do the pauli decomposition of A
    pauli_A = SparsePauliOp.from_operator(Operator(A),rtol=1e-12)
    coeffs = np.real(pauli_A.coeffs)
    labels = pauli_A.paulis.to_labels()

    if len(coeffs) == 0:
        raise ValueError("Pauli decomposition find the coeffs is empty,please check the tolerance of the Pauli decomposition")
    
    # so e^(-iAt) = [e^(-i c_1 P_1 t/r) e^(-i c_2 P_2 t/r) ... e^(-i c_r P_r t/r)]^r

    n_qubits = int(np.log2(A.shape[0]))
    qc = QuantumCircuit(n_qubits)
    for r_i in range(r):
        for k in range(len(labels)):
            label_i = labels[k]
            theta_i = coeffs[k] * t / r
            single = SparsePauliOp([label_i], [theta_i])
            qc.append(PauliEvolutionGate(single, time=1.0), range(n_qubits))
    return qc
            
n = 8
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = -2
    if i + 1 < n:
        A[i, i + 1] = 1
        A[i + 1, i] = 1 
qc3 = Trotter_1st_order(A, t=1.0, r=3)

qc_decomp = qc3.decompose(reps=3)
print(qc_decomp.draw(output='mpl'))
plt.show()