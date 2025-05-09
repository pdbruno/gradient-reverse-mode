"""A class to compute gradients of expectation values."""

from functools import reduce
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit
from .split_circuit import split
from .gradient_lookup import analytic_gradient


class StateGradient:
    """A class to compute gradients of expectation values."""

    def __init__(self, operator: Operator, ansatz: QuantumCircuit, state_in: Statevector, target_parameters=None):
        """
        Args:
            operator (OperatorBase): The operator in the expectation value.
            ansatz (QuantumCircuit): The ansatz in the expecation value.
            state_in (Statevector): The initial, unparameterized state, upon which the ansatz acts.
            target_parameters (List[Parameter]): The parameters with respect to which to derive.
                If None, the derivative for all parameters is computed (also bound parameters!).
        """
        self.operator = operator
        self.ansatz = ansatz
        self.state_in = state_in
        self.target_parameters = target_parameters

        if isinstance(ansatz, QuantumCircuit):
            if type(ansatz) is not QuantumCircuit:
                self.ansatz = ansatz.decompose()

            if target_parameters is None:
                self.unitaries = split(self.ansatz)
                self.paramlist = None
            else:
                self.unitaries, self.paramlist = split(self.ansatz, target_parameters,
                                                       separate_parameterized_gates=False,
                                                       return_parameters=True)
        elif isinstance(ansatz, list):
            self.unitaries = ansatz
            self.paramlist = None
        else:
            raise NotImplementedError('Unsupported type of ansatz.')

    def reference_gradients(self, parameter_binds=None, return_parameters=False):
        op, ansatz, init = self.operator, self.ansatz, self.state_in

        unitaries, paramlist = self.unitaries, self.paramlist
        num_parameters = len(unitaries)

        if paramlist is not None and parameter_binds is None:
            raise ValueError('If you compute the gradients with respect to a ansatz with free '
                             'parameters, you must pass a dictionary of parameter binds.')

        if parameter_binds is None:
            parameter_binds = {}
            paramlist = [[None]] * num_parameters
        else:
            ansatz = _bind(self.ansatz, parameter_binds)

        bound_unitaries = _bind(unitaries, parameter_binds)

        # lam = reduce(lambda x, y: x.evolve(y), ulist, self.state_in).evolve(self.operator)
        lam = self.state_in.evolve(ansatz).evolve(op)
        grads = []
        for j in range(num_parameters):
            grad = 0

            deriv = analytic_gradient(unitaries[j], paramlist[j][0])
            for _, gate in deriv:
                _bind(gate, parameter_binds, inplace=True)

            for coeff, gate in deriv:
                dj_unitaries = bound_unitaries[:max(0, j)] + [gate] \
                    + bound_unitaries[min(num_parameters, j + 1):]
                phi = reduce(lambda x, y: x.evolve(y), dj_unitaries, init)
                grad += coeff * lam.conjugate().data.dot(phi.data)
            grads += [2 * grad.real]

        if parameter_binds == {}:
            return grads

        accumulated, unique_params = self._accumulate_product_rule(grads)
        if return_parameters:
            return accumulated, unique_params

        return accumulated

    def iterative_gradients(self, parameter_binds=None, return_parameters=False):
        op, ansatz, init = self.operator, self.ansatz, self.state_in

        ulist, paramlist = self.unitaries, self.paramlist
        num_parameters = len(ulist)

        if paramlist is not None and parameter_binds is None:
            raise ValueError('If you compute the gradients with respect to a ansatz with free '
                             'parameters, you must pass a dictionary of parameter binds.')

        if parameter_binds is None:
            parameter_binds = {}
            paramlist = [[None]] * num_parameters
        else:
            ansatz = _bind(ansatz, parameter_binds)

        phi = init.evolve(ansatz)
        lam = phi.evolve(op)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = ulist[j]

            deriv = analytic_gradient(uj, paramlist[j][0])
            for _, gate in deriv:
                _bind(gate, parameter_binds, inplace=True)

            uj_dagger = _bind(uj, parameter_binds).inverse()

            phi = phi.evolve(uj_dagger)

            # TODO use projection
            grad = 2 * sum(coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                           for coeff, gate in deriv).real
            grads += [grad]

            if j > 0:
                lam = lam.evolve(uj_dagger)

        if parameter_binds == {}:
            return list(reversed(grads))

        accumulated, unique_params = self._accumulate_product_rule(
            list(reversed(grads)))
        if return_parameters:
            return accumulated, unique_params

        return accumulated

    def _accumulate_product_rule(self, gradients):
        grads = {}
        for paramlist, grad in zip(self.paramlist, gradients):
            # all our gates only have one single parameter
            param = paramlist[0]
            grads[param] = grads.get(param, 0) + grad

        return list(grads.values()), list(grads.keys())


# pylint: disable=inconsistent-return-statements
def _bind(circuits, parameter_binds, inplace=False):
    if not isinstance(circuits, list):
        existing_parameter_binds = {p: parameter_binds[p] for p in circuits.parameters}
        return circuits.assign_parameters(existing_parameter_binds, inplace=inplace)

    bound = []
    for circuit in circuits:
        existing_parameter_binds = {
            p: parameter_binds[p] for p in circuit.parameters}
        bound.append(circuit.assign_parameters(
            existing_parameter_binds, inplace=inplace))

    if not inplace:
        return bound
