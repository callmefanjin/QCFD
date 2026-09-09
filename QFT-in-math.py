import numpy as np

def qft_matrix(n_qubits):
    N = 2**n_qubits
    F = np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            F[j,k] = np.exp(2*np.pi*1j*j*k/N) / np.sqrt(N)
    return F

# check the matrix is unitary

n_qubits = 2
N = 2**n_qubits
F = qft_matrix(n_qubits)
result = F @ F.conj().T

print(np.allclose(result, np.eye(N)), "\n")

# provide basis state vector and apply QFT to it
state_vector_basis = np.array([0.0, 1.0, 0.0, 0.0])
state_vector_basis = state_vector_basis / np.linalg.norm(state_vector_basis)
qft_result = F @ state_vector_basis
print("vector length:", np.linalg.norm(state_vector_basis))
print("before QFT:", state_vector_basis)
print("after QFT:", qft_result)
print("\n")

# try to apply inverse QFT to the result, thus, we need to know the inverse matrix of QFT
# but we should realize that the inverse of F is actually the conjugate transpose of F
inverse_F = F.conj().T
inverse_qft_result = inverse_F @ qft_result
print(
    "Recovered correctly:",
    np.allclose(inverse_qft_result, state_vector_basis)
)
print("\n")



# provide a complex state vector and apply QFT to it
complex_state_vector = np.array([1.0+1.0j, 2.0+2.0j, 2.0, 1.0])
complex_state_vector = complex_state_vector / np.linalg.norm(complex_state_vector)
qft_result = F @ complex_state_vector
print("vector length:", np.linalg.norm(complex_state_vector))
print("before QFT:", complex_state_vector)
print("after QFT:", qft_result)


# we should realize that the QFT is actually a unitary matrix 
# QFT matrix has a specific structure which only depends on the number of qubits
# QFT matrix can not only apply to the basis states, but also apply to any state vector