import numpy as np
from qiskit.circuit import QuantumCircuit
from numba import njit
from qiskit.quantum_info import Statevector

class customStateVector():
    def __init__(self, bitstring: str):
        self.state = np.array([int(c) + 0j for c in bitstring]) #solo 0s y 1s
        self.shape = self.state.shape[0]

    def evolve(self, qc : np.ndarray):
        self.state = prueba_numba(qc, self.state)
    
@njit()    
def prueba_numba(qc : np.ndarray, state : np.ndarray):
    return np.dot(qc, state)
    
if __name__ == '__main__':
    op = np.array([[1,0,0],[0,1,0],[0,0,21]], dtype=np.complex128)
    csv = customStateVector('011')
    csv.evolve(op)
    sv = Statevector(csv.state) 

    

