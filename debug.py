#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:04:57 2023

@author: eeltink
"""

from sympy.physics.quantum import Commutator, Dagger, Operator
import sympy
from sympy import *
from IPython.display import display
import numpy as np

U, F, delta_a, delta_b, J, kappa_a,eta = symbols(r'U F \Delta_a \Delta_b J \kappa_a \eta')

a = Symbol(r'a', commutative=False)
ad = Symbol(r'a^\dag', commutative=False)
b = Symbol(r'b', commutative=False)
bd = Symbol(r'b^\dag', commutative=False)

rho = Symbol(r'\rho', commutative=False)


#%% Consruct hamiltonian, jump, lindblad

Ham1 = J*(ad*b+a*bd) + F*(ad+a) -delta_a*ad*a -delta_b*bd*b  + Rational(1, 2)*U*ad*ad*a*a
Ham2 = -delta_a*ad*a -delta_b*bd*b
Ham3 = Mul(U,a,a,evaluate=False)
Ham4 = UnevaluatedExpr(Mul(Rational(1, 2)*U,ad,ad,a,a,evaluate=False))


MyHam =Ham3
display(MyHam)
# jump


# def getJump(constant,c,rho):
#     cd = Dagger(c)
#     return UnevaluatedExpr(constant*(cd*rho*c - Rational(1, 2)*(c*cd*rho + rho*c*cd)))

# D1 = getJump(kappa_a,a,rho)
D1 = kappa_a*(ad*rho*a - Rational(1, 2)*(a*ad*rho + rho*a*ad))
D2 = eta*(a*a*rho*ad*ad - Rational(1, 2)*(ad*ad*a*a*rho + rho*ad*ad*a*a))
#display(D1)

# Lindblad

ME_comm = -sympy.I* (Commutator(MyHam, rho))+0

display(ME_comm)
ME= ME_comm.doit()
#display(ME)
ME = expand(ME)
display(ME)


#%% Convert to phase space function
from phasespaceconversion import pow_to_mul_sep2,PhaseSpaceFunction
# wigner:
myWigner = PhaseSpaceFunction(ME,'W')
W_eqn = myWigner.getFPfromME()
print('Result')
display(expand(W_eqn))
