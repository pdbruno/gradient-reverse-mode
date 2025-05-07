"""Script to run the benchmarks."""

# pylint: disable=invalid-name

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit
from gradients import StateGradient
from qiskit.circuit.library import EfficientSU2

circuit = EfficientSU2(2).assign_parameters(np.random.random(16))
state_in = Statevector.from_label('00')

grad = StateGradient(Operator.from_label("HH"), circuit, state_in, None)
asd = grad.iterative_gradients()
asd2= grad.reference_gradients()
print(asd, asd2)