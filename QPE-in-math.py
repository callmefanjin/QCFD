# This is a script to understand the QPE algorithm in mathematics

# ============================================================

import numpy as np

# ============================================================
# let's first design the unitary matrix U

phi_exact = 5 / 8
lambda_exact = np.exp(2 * np.pi * 1j * phi_exact)

ket_u = np.array([
    [3/5],
    [4/5]
], dtype=complex)

ket_v = np.array([
    [-4/5],
    [ 3/5]
], dtype=complex)

U = (
    ket_v @ np.conj(ket_v).T
    + lambda_exact * ket_u @ np.conj(ket_u).T
)

# so here is a eigenvalue problem:
# U |u> = lambda |u>
# and lambda could be written as:
# lambda = e^(2 * pi * 1j * phi)
# where phi is the phase of the eigenvalue

# so the question is: how to get phi from U and |u>?

# ============================================================

# before solving this problem in quantum computing,
# let's solve it in classical computing first

# we know that:
# U |u> = lambda |u>
# so we can write:
# <u| U |u> = lambda <u|u> = lambda

lambda_in_math = np.dot(np.conj(ket_u).T, U @ ket_u)
phase_in_math = np.angle(lambda_in_math)/(2 * np.pi)
print(f"phase in math: {phase_in_math}" "\n")  

# here we will get the value is -3/8(-0.375),
# which also means that the phase value could be written as 5/8 considering the periodic property.

# ============================================================

# so how to get the phase value in quantum computing?
# in quantum computing, we can only use those quantum gates,
# which means we should find the phase value in the process of applying those quantum gates.
# additionally, we can only measure the probability of the state, 
# so transferring the phase value to the probability is a good idea.

# go back to the beginning, we have the unitary matrix U and the eigenvector |u>.
# naturally, why not applying the unitary matrix U to the eigenvector |u> directly:
U_ket_u = U @ ket_u
print(f"U|u>: {U_ket_u}")

# here we will get the result is:
# U @ ket_u: [[-0.42426407-0.42426407j]
#             [-0.56568542-0.56568542j]]

# one thing to notice is that:
# the two basis states are |0> = [1, 0] and |1> = [0, 1]
# if we measure the ket_u:
P_ket0 = np.abs(ket_u[0])**2
P_ket1 = np.abs(ket_u[1])**2
print(f"P(|0>): {P_ket0}, P(|1>): {P_ket1}")

# and if we measure the U_ket_u:
P_U_ket0 = np.abs(U_ket_u[0])**2
P_U_ket1 = np.abs(U_ket_u[1])**2
print(f"P(U|0>): {P_U_ket0}, P(U|1>): {P_U_ket1}" "\n")

# we will find that:
# P(|0>): [0.36], P(|1>): [0.64]
# P(U|0>): [0.36], P(U|1>): [0.64]
# That means: 
# we cannot get the phase or lambda value from the definition: U |u> = lambda |u> directly.

# ============================================================

# since a single qubit is not engouh to express the phase value, 
# we can introduce an auxiliary qubit to detect the phase value.

ket_auxiliary = np.array([[1],[0]])

# the whole state has become: |0> ⊗ |u>
# apply hadamard gate to the auxiliary qubit: (H ⊗ I) (|0> ⊗ |u>) = H|0> ⊗ I|u>

H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
H_ket0 = H_gate @ ket_auxiliary
print(f"H|0>: {H_ket0}" "\n")

# we will get the result is:
# H|0>: [[1/sqrt(2)], [1/sqrt(2)]]
# which means that the auxiliary qubit is in the superposition state:
# |0> + |1> / sqrt(2)
# and the whole state has become: (1/sqrt(2)) (|0> + |1>) ⊗ |u>

print(f"whole state in math vector: {np.kron(H_ket0, ket_u)}" "\n")

# since we have (1/sqrt(2)) (|0> + |1>) ⊗ |u>, how to get the phase value or lambda from this state?
# one thing should be noticed is that: U |u> = lambda |u>
# we have know that it is useless to apply the U to the whole state, 
# so we can apply the U to the ket_u only like this: 
# (1/sqrt(2)) (|0>|u> + |1>|u>)  ---> (1/sqrt(2)) (|0>|u> + |1>U|u>) ---> (1/sqrt(2)) (|0>|u> + |1>lambda|u>)
# --->(1/sqrt(2)) (|0>|u> + |1> e^(2 * pi * 1j * phi) |u>)
# here is the most interesting part: a quantum gate U become a number lambda 
# but the problem remains,lambda = e^(2 * pi * 1j * phi) could not be reflected in the probability of the auxiliary qubit.
# because e^(2 * pi * 1j * phi) is actually unit vector
# phi is only a phase imformation and we need to transfer it to amplitude information because the probability is the amplitude squared.

# so naturally, from amplitude to phase, that is Fourier transform and inverse Fourier transform!
# or more specifically, quantum Fourier transform and inverse quantum Fourier transform!

# ============================================================

# let's recall the definition of the quantum Fourier transform:
# QFT |x> = 1/sqrt(N) Σ_{k=0}^{N-1} e^(2 * pi * 1j * k * x / N) |k>
# and the inverse quantum Fourier transform is:
# IQFT |x> = 1/sqrt(N) Σ_{k=0}^{N-1} e^(-2 * pi * 1j * k * x / N) |k>

# to use this, it is easy to notice that (1/sqrt(2)) (|0>|u> + |1> e^(2 * pi * 1j * phi) |u>) is not enough.

# considering introducing more qubits to the system, if introduce two qubits, the state will be:
# |0| ⊗ |0> ⊗ |u> 
# applying Hadamard gate to all auxiliary qubits, the state will be:
# (1/sqrt(2)) (|0> + |1>) ⊗ (1/sqrt(2)) (|0> + |1>) ⊗ |u>  ---> (1/2) (|00> + |01> + |10> + |11>) ⊗ |u>
# then remember that |k> is a binary representation,
# apply U according to the number of |k>, for example:
# |00>⊗|u> do not aplly U, |01>⊗|u> apply U, |10>⊗|u> apply U^2, |11>⊗|u> apply U^3
# the pattern is that: |k>⊗|u> apply U^(k)
# so (1/2) (|00> + |01> + |10> + |11>) ⊗ |u>  ---> (1/2) (|00> + |01> U|u> + |10> U^2|u> + |11> U^3|u>)
# we should still remember that: U |u> = lambda |u>
# so (1/2) (|00> |u> + |01> U|u> + |10> U^2|u> + |11> U^3|u>)  ---> (1/2) (|00> |u> + |01> lambda|u> + |10> lambda^2|u> + |11> lambda^3|u>)
# ---> (1/2) (|00> + |01> e^(2 * pi * 1j * phi) |u> + |10> e^(2 * pi * 1j * 2 * phi) |u> + |11> e^(2 * pi * 1j * 3 * phi) |u>)
# so similarly, if we have n auxiliary qubits, the state will be:
# (1/sqrt(2^n)) (|00...0> + |00...1> e^(2 * pi * 1j * phi) + |00...10> e^(2 * pi * 1j * 2 * phi) + ... + |11...1> e^(2 * pi * 1j * (2^n-1) * phi)) ⊗ |u>
# = (1/sqrt(2^n)) Σ_{k=0}^{2^n-1} e^(2 * pi * 1j * k * phi) |k> ⊗ |u> = (1/sqrt(N)) Σ_{k=0}^{N-1} e^(2 * pi * 1j * k * phi) |k> ⊗ |u>

# IQFT apply to the state above:
# IQFT (1/sqrt(N)) Σ_{k=0}^{N-1} e^(2 * pi * 1j * k * phi) |k> ⊗ |u> = (1/N) Σ_{k=0}^{N-1}Σ_{x=0}^{N-1} e^(2 * pi * 1j * (k * phi - k * x / N)) |x> ⊗ |u>


# till now, the beginning state: |0>^n ⊗ |u> has become: (1/N) Σ_{k=0}^{N-1}Σ_{x=0}^{N-1} e^(2 * pi * 1j * (k * phi - k * x / N)) |x> ⊗ |u>
# which it has been through Hadamard gate to every auxiliary qubit, controling gate U^(k)  and finally inverse quantum Fourier gate.
# and more importantly, if we measure |x> right now, the probability of |x> is:
# P(|x>) = | (1/N) Σ_{k=0}^{N-1} e^(2 * pi * 1j * (k * phi - k * x / N)) |^2
# then we can get the prpbability distribution of |x>.

# ============================================================\
# rewrite the probability distribution of |x>:
# P(|x>) = 1/(N^2) |(1 + e^(2 * pi * 1j * (phi - x / N)) + e^(2 * pi * 1j * 2* (phi - x / N)) + ... + e^(2 * pi * 1j * (N-1)* (phi - x / N)))| ^2

# consider the latter part of the equation as the sum of a series vector, so when the sum would be the most big since every vector is a unit vector.
# the answer is that: when phi = x / N, the probability is the most big.

# so the most exciting thing is that: if we have the probability distribution of |x> and we will know which |x> has the most big probability
# and naturally, we can calculate the phi value from the x value.

