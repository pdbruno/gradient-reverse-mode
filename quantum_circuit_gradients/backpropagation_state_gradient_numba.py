"""A class to compute gradients of expectation values."""

import concurrent.futures
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit
from .split_circuit import split
from .gradient_lookup import analytic_gradient
from numpy.typing import NDArray
from numba import njit
import os
import numpy as np

class BackpropagationStateGradient_Numba:
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
    
    def step_derivs(self, parameter_binds: NDArray, ulist, paramlist):
        num_parameters = len(ulist)
        step_j_derivs = {}
        for j in reversed(range(num_parameters)):
            uj = ulist[j]
            deriv = analytic_gradient(uj, paramlist[j][0])
            for _, gate in deriv:
                self._bind(gate, parameter_binds, inplace=True)
            deriv = [[step[0], Operator(step[1])._data.astype(np.complex64)] for step in deriv]
            step_j_derivs[j] = deriv
        return step_j_derivs

    def gradients_single(self, parameter_binds: NDArray):
        op, ansatz = self.operator, self.ansatz
        ulist, paramlist = self.unitaries, self.paramlist
        num_parameters = len(ulist)
        ansatz: QuantumCircuit = self._bind(ansatz, parameter_binds)  
        phi = Statevector.from_label("0"*ansatz.num_qubits).evolve(ansatz)
        e = phi.expectation_value(op)
        lam = phi.evolve(op)
        in_training_loop = (os.environ.get("ESTADO_GLOBAL_EN_ENTRENAMIENTO", "True") == "True")
        if in_training_loop:
            grads = []
            # Calculamos de antemano las cosas not numba friendly, y las pasamos a matrices
            step_derivs = self.step_derivs(parameter_binds, ulist, paramlist)
            step_j_dagger = {self._bind(uj, parameter_binds).inverse() for j in range(num_parameters)}
            step_j_dagger_matrix = {Operator(step_j_dagger[j])._data.astype(np.complex64) for j in range(num_parameters)}
            phi_np_probabilities = phi.probabilities().astype(np.complex64)
            lam_np_probabilities = lam.probabilities().astype(np.complex64)
            grads = self.numba_magia_negra(step_derivs, step_j_dagger_matrix, phi_np_probabilities, lam_np_probabilities)
            accumulated, unique_params = self._accumulate_product_rule(list(reversed(grads)))
        return e, ([accumulated[unique_params.index(p)] for p in list(self.ansatz.parameters)] if in_training_loop else None)

    def gradients(self, parameter_binds: NDArray):
        expectation_values = []
        grads = []
        map_func = map
        for e, grad in map_func(self.gradients_single, parameter_binds):
            expectation_values.append(e)
            if grad != None:
                grads.append(grad)
        if len(grads) == 0:
            grads = None
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

    @staticmethod
    @njit(nopython=True)
    def numba_magia_negra(step_derivs: [[np.complex64, NDArray]], step_j_matrix_dagger: NDArray,phi: NDArray, lam: NDArray):
        for j in reversed(range(num_parameters)):
            uj_dagger = step_j_matrix_dagger[j]
            phi = uj_dagger @ phi
            deriv = step_derivs[j]
            grad = 0
            for coeff, matrix in deriv:
                lam_conjugate = np.conjugate(lam)
                phi_evolved = matrix @ phi
                grad += coeff * (lam_conjugate @ phi_evolved)
            grad = (2 * grad).real
            grads += [grad]
            if j > 0:
                lam = uj_dagger @ lam
        return grads

