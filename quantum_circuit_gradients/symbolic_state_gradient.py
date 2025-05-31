from sympy import Matrix, factor_terms, signsimp, lambdify, trigsimp
from sympy.simplify.simplify import sum_simplify, product_simplify
from qiskit.quantum_info import Operator
from qiskit.circuit import QuantumCircuit
from qiskit_symb.quantum_info import Statevector as StatevectorSymb
from numpy.typing import NDArray
import numpy as np


def my_simplify(e):
    e = signsimp(e)
    e = factor_terms(e, sign=False)
    e = trigsimp(e)

    e = sum_simplify(e)
    return product_simplify(e).doit()
    """ 
    short = shorter(e, factor_terms(e), expand_power_exp(expand_mul(e)))

    # get rid of hollow 2-arg Mul factorization
    hollow_mul = Transform(
        lambda x: Mul(*x.args),
        lambda x:
        x.is_Mul and
        len(x.args) == 2 and
        x.args[0].is_Number and
        x.args[1].is_Add and
        x.is_commutative)
    e = short.xreplace(hollow_mul).doit()

    
           """


class SymbolicStateGradient:
    """A class to compute gradients of expectation values."""

    def __init__(self, operator: Operator, ansatz: QuantumCircuit):
        op = Matrix(operator.to_matrix())
        phi = StatevectorSymb(ansatz).to_sympy().reshape(
            2**ansatz.num_qubits, 1).tomatrix()
        #phi = my_simplify(phi)

        e = (phi.adjoint() * op * phi)[0, 0]
        #e = my_simplify(e)
        parameters = [p.sympify() for p in ansatz.parameters]

        self.lambdified_expectation_value = lambdify(
            [parameters], e, modules="numpy")

        self.lambdified_gradient = lambdify(
            [parameters], [e.diff(p) for p in parameters], modules="numpy")

    def gradients(self, parameter_binds: NDArray):
        return self.lambdified_expectation_value(parameter_binds.T), -np.array(self.format_gradients(parameter_binds)).T

    def format_gradients(self, parameter_binds):
        length = len(parameter_binds)
        return [self.vectorize_0(partial_derivative, length) for partial_derivative in self.lambdified_gradient(parameter_binds.T)]

    def vectorize_0(self, partial_derivative, length): #segurisimo existe una forma mas eficiente de arreglar el problema de los 0s
        return partial_derivative if type(partial_derivative) != int else np.zeros(length)
