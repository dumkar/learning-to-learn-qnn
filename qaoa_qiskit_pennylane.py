import itertools
import numpy as np
from functools import partial, reduce
from qiskit import *# BasicAer, QuantumRegister, execute
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator, get_aer_backend
from qiskit.aqua.components.initial_states import Custom
from scipy.optimize import minimize
from qiskit.compiler import transpile, assemble
from qiskit.transpiler import CouplingMap, PassManager, Layout
from qiskit.transpiler.passmanager import DoWhileController, ConditionalController, FlowController
from qiskit.transpiler.passes import *
from qiskit.extensions.standard import *
import pennylane as qml
from pennylane import numpy as np
np.set_printoptions(precision=3, suppress=True)

import re

def pauli_x(qubit, coeff,n_qubits):
    eye = np.eye((n_qubits))
    return Operator([[coeff, Pauli(np.zeros(n_qubits), eye[qubit])]])
def pauli_z(qubit, coeff,n_qubits):
    eye = np.eye((n_qubits))
    return Operator([[coeff, Pauli(eye[qubit], np.zeros(n_qubits))]])

def product_pauli_z(q1, q2, coeff,n_qubits):
    eye = np.eye((n_qubits))
    return Operator([[coeff, Pauli(eye[q1], np.zeros(n_qubits)) * Pauli(eye[q2], np.zeros(n_qubits))]])

def evolve(hamiltonian, angle, quantum_registers):
    return hamiltonian.evolve(None, angle, 'circuit', 1,
                              quantum_registers=quantum_registers,
                              expansion_mode='suzuki',
                              expansion_order=3)

def create_circuit(beta, gamma,p,Hm,Hc,qr):
    circuit_evolv = reduce(lambda x,y: x+y, [evolve(Hc, beta[i], qr) + evolve(Hm, gamma[i], qr)
                                             for i in range(p)])
    return circuit_evolv

# set p beforehand
p = 2

# dev = qml.device('default.qubit', wires=2)

# @qml.qnode(dev, interface='torch')
def QAOA_circ(parameters):# = np.random.uniform(0, np.pi*2, 2*p)):
    
    beta = parameters[:p]
    gamma = parameters [p:]
    
    def set_up_QAOA_in_IBM(beta, gamma, p , n_qubits = 2, J = np.array([[0,1],[0,0]])):
        
        Hm = reduce(lambda x, y: x+y,[pauli_x(i, 1,n_qubits) for i in range(n_qubits)])
        Hm.to_matrix()
        Hc = reduce(
            lambda x,y:x+y,
            [product_pauli_z(i,j, -J[i,j],n_qubits) for i,j in itertools.product(range(n_qubits), repeat=2)]
        )
        Hc.to_matrix()
        init_state_vect = [1 for i in range(2**n_qubits)]
        init_state = Custom(n_qubits, state_vector=init_state_vect)
        qr = QuantumRegister(n_qubits)
        circuit_init = init_state.construct_circuit('circuit', qr)
        qaoa_circuit = create_circuit(beta, gamma,p,Hm,Hc,qr)
        qaoa_circuit=qaoa_circuit+circuit_init
        
        return qaoa_circuit
    
    qc=set_up_QAOA_in_IBM(beta, gamma, p)
    
    
    qaoa_circuit = transpile(qc,
                            seed_transpiler=11,
                            optimization_level=3)
    
    a=qaoa_circuit.qasm()
    
    IBM_gate_list=re.sub('q[0-9]+','', a).split('\n')  

    for item in IBM_gate_list:
        u_p_1=None
        q_1=None
        q_2=None
        u_p_2=None
        u_p_3=None
        if 'u1' in item: 
            temp=item.replace('u1(','')
            u_p_1=temp[:temp.find(')')]
            q_1=temp[temp.find(']')-1] 
            qml.PhaseShift(float(u_p_1), wires=q_1)
        elif 'u2' in item:
            temp=item.replace('u2(','')
            u_p_1=temp[temp.find(',')-1]
            u_p_2=temp[temp.find(',')+1:temp.find(')')]
            q_1=temp[temp.find(']')-1]
            qml.Rot(float(u_p_1),np.pi,float(u_p_2),wires=q_1)
        elif 'u3' in item:
            temp=item.replace('u3(','')
            u_p_1=temp[:temp.find(',')]
            s1=temp.find(',',1)
            s2=temp.find(',',s1+1)
            u_p_2=temp[s1+1:s2]
            u_p_3=temp[s2+1:temp.find(')')]
            q_1=temp[temp.find(']')-1]
            qml.Rot(float(u_p_1),float(u_p_3),float(u_p_2),wires=q_1)
        elif 'cx' in item:
            temp=item.replace('cx','')
            q_1=temp[temp.find('[',1)+1]
            q_2=temp[temp.find('[',2)+1]
            qml.CNOT(wires=[q_1, q_2])
    return qml.expval(qml.PauliZ(0)) #fix this