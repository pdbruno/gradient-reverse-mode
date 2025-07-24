"""A class to compute gradients of expectation values."""

import concurrent.futures
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import QuantumCircuit
from .split_circuit import split
from .gradient_lookup import analytic_gradient
from numpy.typing import NDArray
import os

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
                                               separate_parameterized_gates=True)

    def gradients_single(self, parameter_binds: NDArray):
        #parameter_binds.shape == 2 * entity_embedding_dim + relation_embedding_dim. Es un batch concatenado
        # op es el overlap_estimator operator, y el ansatz es del circuito completo
        op, ansatz = self.operator, self.ansatz

        ulist, paramlist = self.unitaries, self.paramlist
        num_parameters = len(ulist)

        ansatz: QuantumCircuit = self._bind(ansatz, parameter_binds)  # type: ignore

        phi = Statevector.from_label("0"*ansatz.num_qubits).evolve(ansatz)
        e = phi.expectation_value(op).real
        lam = phi.evolve(op)
        in_training_loop = (os.environ.get("ESTADO_GLOBAL_EN_ENTRENAMIENTO", "True") == "True")
        if in_training_loop:
            #print(" Calculo del gradiente analitico")
            grads = []
            for j in reversed(range(num_parameters)):
                uj = ulist[j]

                deriv = analytic_gradient(uj, paramlist[j][0] if len(paramlist[j]) == 1 else None)
                for _, gate in deriv:
                    self._bind(gate, parameter_binds, inplace=True)

                uj_dagger = self._bind(uj, parameter_binds).inverse()

                phi = phi.evolve(uj_dagger)

                # TODO use projection
                grad = 0
                for coeff, gate in deriv:
                    grad += coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                grad = (2 * grad).real
                grads += [grad]

                if j > 0:
                    lam = lam.evolve(uj_dagger)

            accumulated, unique_params = self._accumulate_product_rule(
                list(reversed(grads)))
        ret_snd = ([accumulated[unique_params.index(p)] for p in list(self.ansatz.parameters)] if in_training_loop else None)
        return e, ret_snd

    def gradients(self, parameter_binds: NDArray):
        # parameter_binds son los embeddings h,r,t concatenados. Vienen batcheados de a 3 por default
        expectation_values = []
        grads = []
        map_func = map
        for e, grad in map_func(self.gradients_single, parameter_binds): #gradients_single por cada batch
            expectation_values.append(e)
            if grad != None:
                grads.append(grad)
        if len(grads) == 0:
            grads = None
        #print(expectation_values)
        return expectation_values, grads

    def _accumulate_product_rule(self, gradients):
        grads = {}
        for paramlist, grad in zip(self.paramlist, gradients):
            # all our gates only have one single parameter
            if len(paramlist) == 0:
                continue
            param = paramlist[0]
            grads[param] = grads.get(param, 0) + grad

        return list(grads.values()), list(grads.keys())


    def _bind(self, circuit: QuantumCircuit, parameter_binds: NDArray, inplace=False):
        parameter_indexes = [self.ansatz.parameters.data.index(p) for p in circuit.parameters]
        return circuit.assign_parameters(parameter_binds[parameter_indexes], inplace=inplace)
