"""A class to compute gradients of expectation values."""

import concurrent.futures
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit
from .split_circuit import split
from .gradient_lookup import analytic_gradient
from numpy.typing import NDArray


class BackpropagationStateGradient:
    """A class to compute gradients of expectation values."""

    def __init__(self, operator: Operator, ansatz: QuantumCircuit):
        """
        Args:
            operator (OperatorBase): The operator in the expectation value.
            ansatz (QuantumCircuit): The ansatz in the expecation value.
            state_in (Statevector): The initial, unparameterized state, upon which the ansatz acts.
        """
        self.operator = operator
        self.ansatz = ansatz

        self.unitaries, self.paramlist = split(self.ansatz, list(ansatz.parameters),
                                               separate_parameterized_gates=False)

    def gradients_single(self, parameter_binds: NDArray):
        op, ansatz = self.operator, self.ansatz

        ulist, paramlist = self.unitaries, self.paramlist
        num_parameters = len(ulist)

        ansatz: QuantumCircuit = self._bind(ansatz, parameter_binds)  # type: ignore

        phi = Statevector.from_label("0"*ansatz.num_qubits).evolve(ansatz)
        lam = phi.evolve(op)
        e = phi.expectation_value(op)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = ulist[j]

            deriv = analytic_gradient(uj, paramlist[j][0])
            for _, gate in deriv:
                self._bind(gate, parameter_binds, inplace=True)

            uj_dagger = self._bind(uj, parameter_binds).inverse()

            phi = phi.evolve(uj_dagger)

            # TODO use projection
            grad = 2 * sum(coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                           for coeff, gate in deriv).real
            grads += [grad]

            if j > 0:
                lam = lam.evolve(uj_dagger)

        accumulated, unique_params = self._accumulate_product_rule(
            list(reversed(grads)))

        return e, [accumulated[unique_params.index(p)] for p in list(self.ansatz.parameters)]

    def gradients(self, parameter_binds: NDArray):
        expectation_values = []
        grads = []
        batch_size = len(parameter_binds)
        if batch_size <= 300:
            map_func = map
        else:
            executor = concurrent.futures.ProcessPoolExecutor()
            map_func = executor.map

        for e, grad in map_func(self.gradients_single, parameter_binds):
            expectation_values.append(e)
            grads.append(grad)

        if batch_size > 300:
            executor.shutdown(wait=True)

        return expectation_values, grads

    def _accumulate_product_rule(self, gradients):
        grads = {}
        for paramlist, grad in zip(self.paramlist, gradients):
            # all our gates only have one single parameter
            param = paramlist[0]
            grads[param] = grads.get(param, 0) + grad

        return list(grads.values()), list(grads.keys())


    def _bind(self, circuit: QuantumCircuit, parameter_binds: NDArray, inplace=False):
        parameter_indexes = [self.ansatz.parameters.data.index(p) for p in circuit.parameters]
        return circuit.assign_parameters(parameter_binds[parameter_indexes], inplace=inplace)
