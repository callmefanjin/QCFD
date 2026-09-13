# a tool code for LCU which is used to any possible matrix A
# ===============================================================
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
# ===============================================================

def LCU(A, psi):
    # check if A is a zero matrix
    if np.all(A == 0):
        raise ValueError("A is a zero matrix")

    N = A.shape[0]
    # first check N is a power of 2
    if N  & (N - 1) != 0:
        nqubits_system = int(np.ceil(np.log2(N)))
        A_reconstructed = np.zeros((2**nqubits_system, 2**nqubits_system), dtype=complex)
        psi_reconstructed = np.zeros(2**nqubits_system, dtype=complex)
        A_reconstructed[:N, :N] = A
        psi_reconstructed[:N] = psi
    else:
        nqubits_system = int(np.log2(N))
        A_reconstructed = A
        psi_reconstructed = psi

    # Pauli decomposition of A_reconstructed
    pauli_A = SparsePauliOp.from_operator(Operator(A_reconstructed), atol=1e-10, rtol=1e-6)
    coeffs = pauli_A.coeffs
    if len(coeffs) == 0:
        raise ValueError("Pauli decomposition find the coeffs is empty,please check the tolerance of the Pauli decomposition")
    s = np.sum(np.abs(coeffs))
    w = np.abs(coeffs)
    phase_factor = coeffs / w  # seperate the phase imformation
    P = pauli_A.paulis.to_matrix()
    U = phase_factor[:,None,None] * P

    nqubits_ancilla = int(np.ceil(np.log2(len(coeffs))))

    # create the quantum circuit
    ancilla_list = list(range(nqubits_ancilla))
    system_list = list(range(nqubits_ancilla, nqubits_system + nqubits_ancilla))
    qc = QuantumCircuit(nqubits_system + nqubits_ancilla)
    # prepare the state |psi>
    qc.append(StatePreparation(psi_reconstructed), system_list)

    if nqubits_ancilla == 0:  # in case of only one term
        qc.append(UnitaryGate(U[0]), system_list)
        return qc, s, ancilla_list, system_list

    # prepare the ancilla qubits
    ancilla_coeffs = np.zeros(2**nqubits_ancilla, dtype=complex)
    ancilla_coeffs[:len(coeffs)] = np.sqrt(w/s)
    ancilla_state_preparation = StatePreparation(ancilla_coeffs)
    qc.append(ancilla_state_preparation, ancilla_list)

    # apply controlled-unitary gates
    for i in range(len(coeffs)):
        gate = UnitaryGate(U[i])

        controlled_gate = gate.control(nqubits_ancilla, ctrl_state=i)
        qc.append(controlled_gate, ancilla_list + system_list)

    # apply inverse ancilla_state_preparation
    qc.append(ancilla_state_preparation.inverse(), ancilla_list)

    return qc, s, ancilla_list, system_list

# ===============================================================

# define the matrix A
N = 2
A = np.random.randn(N, N)
psi = np.random.randn(N)
psi = psi / np.linalg.norm(psi)

qc, s, ancilla_list, system_list = LCU(A, psi)

final_state = Statevector.from_instruction(qc)
success = final_state.data[::2**len(ancilla_list)]
posibility_success = np.sum(np.abs(success)**2)
output_state = success / np.sqrt(posibility_success)






    





    
    



