import itertools
from qiskit.circuit import QuantumCircuit, CircuitInstruction, Instruction, ParameterExpression
from qiskit.circuit.library import (
    RXGate, RYGate, RZGate, CRXGate, CRYGate, CRZGate
)
from typing import cast


def gradient_lookup(gate):
    """Returns a circuit implementing the gradient of the input gate."""

    param: ParameterExpression = gate.params[0]
    param_derivative = param.gradient(list(param.parameters)[0])
    if isinstance(gate, RXGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rx(param, 0)
        derivative.x(0)
        return [[0.5j * param_derivative, derivative]]
    if isinstance(gate, RYGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.ry(param, 0)
        derivative.y(0)
        return [[0.5j * param_derivative, derivative]]
    if isinstance(gate, RZGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rz(param, 0)
        derivative.z(0)
        return [[0.5j * param_derivative, derivative]]
    if isinstance(gate, CRXGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rx(param, 1)
        proj1.x(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rx(param, 1)
        proj2.x(1)

        return [[0.25j * param_derivative, proj1], [-0.25j * param_derivative, proj2]]
    if isinstance(gate, CRYGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.ry(param, 1)
        proj1.y(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.ry(param, 1)
        proj2.y(1)

        return [[0.25j * param_derivative, proj1], [-0.25j * param_derivative, proj2]]
    if isinstance(gate, CRZGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rz(param, 1)
        proj1.z(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rz(param, 1)
        proj2.z(1)

        return [[0.25j * param_derivative, proj1], [-0.25j * param_derivative, proj2]]
    raise NotImplementedError('Cannot implement for', gate)


def analytic_gradient(circuit: QuantumCircuit, parameter=None):
    """Return the analytic gradient of the input circuit."""

    if parameter is not None:
        if parameter not in circuit.parameters:
            raise ValueError('Parameter not in this circuit.')

    summands, op_context = [], []
    for i, op in enumerate(circuit.data):
        gate: Instruction = op[0]
        op_context += [op[1:]]
        if (parameter is None and len(gate.params) > 0) or parameter in [free_param for gate_param in gate.params for free_param in cast(ParameterExpression, gate_param).parameters ]:
            summands += [gradient_lookup(gate)]
        else:
            summands += [[[1, gate]]]

    gradient = []
    for product_rule_term in itertools.product(*summands):
        summand_circuit = QuantumCircuit(*circuit.qregs)
        coeff = 1
        for i, a in enumerate(product_rule_term):
            coeff *= a[0]
            summand_circuit.append(a[1], *op_context[i])
        gradient += [[coeff, summand_circuit.copy()]]

    return gradient
