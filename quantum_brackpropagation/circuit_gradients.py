"""A class to compute gradients of expectation values."""

from functools import reduce
import concurrent.futures
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit, Parameter
from .split_circuit import split
from .gradient_lookup import analytic_gradient
from typing import Union, List, Mapping


class StateGradient:
    """A class to compute gradients of expectation values."""

    def __init__(self, operator: Operator, ansatz: QuantumCircuit, state_in: Statevector):
        """
        Args:
            operator (OperatorBase): The operator in the expectation value.
            ansatz (QuantumCircuit): The ansatz in the expecation value.
            state_in (Statevector): The initial, unparameterized state, upon which the ansatz acts.
        """
        self.operator = operator
        self.ansatz = ansatz
        self.state_in = state_in

        self.unitaries, self.paramlist = split(self.ansatz, list(ansatz.parameters),
                                                    separate_parameterized_gates=False)

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

    def iterative_gradients_single(self, parameter_binds: Mapping[Parameter, float]):
        op, ansatz, init = self.operator, self.ansatz, self.state_in

        ulist, paramlist = self.unitaries, self.paramlist
        num_parameters = len(ulist)
        
        ansatz: QuantumCircuit = _bind(ansatz, parameter_binds) # type: ignore

        phi = init.evolve(ansatz)
        lam = phi.evolve(op)
        e = phi.expectation_value(op)
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

        accumulated, unique_params = self._accumulate_product_rule(
            list(reversed(grads)))
        
        
        return e, [accumulated[unique_params.index(p)] for p in list(parameter_binds.keys())]

    def iterative_gradients(self, parameter_binds: Mapping[Parameter, List[float]]):
        expectation_values = []
        grads = []
        batch_size = len(next(iter(parameter_binds.values())))
        if batch_size <= 300:
            map_func = map
        else:
            executor = concurrent.futures.ProcessPoolExecutor()
            map_func = executor.map

        for e, grad in map_func(self.iterative_gradients_single, 
                            [{p: parameter_binds[p][i].item() for p in parameter_binds} 
                                for i in range(batch_size)
                            ]):
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


# pylint: disable=inconsistent-return-statements
def _bind(circuits: QuantumCircuit, parameter_binds: Mapping[Parameter, float], inplace=False):
    existing_parameter_binds = {p: parameter_binds[p] for p in circuits.parameters}
    return circuits.assign_parameters(existing_parameter_binds, inplace=inplace)
