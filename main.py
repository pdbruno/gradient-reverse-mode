"""Script to run the benchmarks."""

# pylint: disable=invalid-name

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import n_local
from quantum_circuit_gradients import BackpropagationStateGradient
from qiskit.circuit.library import EfficientSU2

circuit = n_local(4,
            ["rx","ry","rz"],
            ["crx","cry", "crz"],
            "full",
            reps=1,
            insert_barriers=False,
            skip_final_rotation_layer=True,
)
circuit.assign_parameters(np.random.random(30))
parameter_binds_input = np.random.random(30)
grad = BackpropagationStateGradient(Operator.from_label("HHHH"), circuit)
res = grad.gradients_single(parameter_binds_input)
print(res)