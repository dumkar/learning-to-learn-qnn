import itertools
from functools import partial
from pyquil import Program, api
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, sZ
from pyquil.gates import *
from scipy.optimize import minimize
#from forest_tools import *
import pennylane as qml
from pennylane import numpy as np
np.set_printoptions(precision=3, suppress=True)

import re

####Pauli matrices
𝜎𝑌=np.array([[0, -1j],[1j, 0]])
𝜎𝑍=np.array([[1, 0],[0, -1]])
𝜎𝑋=np.array([[0, 1],[1, 0]])
𝜎𝐼=np.eye(2)
#####

def prepare_𝛾𝛾(n_qubits,i,j,𝛾):
    #prepares ⋯𝐼⊗𝛾𝑖⊗𝐼⋯⊗𝛾𝑗⊗𝐼⋯ matrix
    𝛾_𝛾=1
    for k in range(n_qubits):
        if k==i or k==j:
            𝛾_𝛾=np.kron(𝛾_𝛾,𝛾)
        else:
            𝛾_𝛾=np.kron(𝛾_𝛾,𝜎𝐼)
    return 𝛾_𝛾
def prepare_𝛾(n_qubits,i,𝛾):
    #prepares ⋯𝐼⊗𝛾𝑖⊗𝐼⋯ matrix
    I_𝛾=1
    for k in range(n_qubits):
        if k==i:
            I_𝛾=np.kron(I_𝛾,𝛾)
        else:
            I_𝛾=np.kron(I_𝛾,𝜎𝐼)
    return I_𝛾

def prepare_qaoa_hamiltonian(J,n_qubits)
    Hamilton=np.zeros([2**n_qubits,2**n_qubits])
    for i in range(n_qubits):
        for k in range(n_qubits):
            if J[i][k] != 0:
                print([i,k])
                if i==k:
                    Hamilton+=J[i][k]*prepare_𝛾(n_qubits,i,𝜎𝑍)
                else:
                    Hamilton+=J[i][k]*prepare_𝛾𝛾(n_qubits,i,k,𝜎𝑍) 
    return Hamilton

def create_circuit(beta, gamma,initial_state,exp_Hm,exp_Hc):
    circuit = Program()
    circuit += initial_state
    for i in range(p):
        for term_exp_Hc in exp_Hc:
            circuit += term_exp_Hc(-beta[i])
        for term_exp_Hm in exp_Hm:
            circuit += term_exp_Hm(-gamma[i])

    return circuit

# set p beforehand
p = 2

# dev = qml.device('default.qubit', wires=2)

# @qml.qnode(dev, interface='torch')
def QAOA_circ(parameters):# = np.random.uniform(0, np.pi*2, 2*p)):
    
    beta = parameters[:p]
    gamma = parameters [p:]
    
    def set_up_QAOA_in_pyquil(beta, gamma, p , n_qubits = 2, J = np.array([[0,1],[0,0]])):
                
        Hm = [PauliTerm("X", i, 1.0) for i in range(n_qubits)]
        
        Hc = []
        initial_state = Program()
        for i in range(n_qubits):
            initial_state += H(i)      
        for i in range(n_qubits):
            for j in range(n_qubits):
                Hc.append(PauliTerm("Z", i, -J[i, j]) * PauliTerm("Z", j, 1.0))
        exp_Hm = []
        exp_Hc = []
        for term in Hm:
            exp_Hm.append(exponential_map(term))
        for term in Hc:
            exp_Hc.append(exponential_map(term)) 
        qaoa_circuit = create_circuit(beta, gamma,initial_state,exp_Hm,exp_Hc)
        
        return qaoa_circuit
    
    pyquil_circ=set_up_QAOA_in_pyquil(beta, gamma, p)
    pyquil_circ_list=str(pyquil_circ).split('\n')
    for item in pyquil_circ_list:
        u_p_1=None
        q_1=None
        q_2=None
        u_p_2=None
        u_p_3=None
        if 'H' in item: 
            q_1=item[item.find('H')+2] 
            qml.Hadamard(wires=q_1)
        elif 'RZ(' in item:
            temp=item.replace('RZ(','')
            u_p_1=temp[:temp.find(')')]
            q_1=temp[temp.find(']')+2]
            qml.RZ(float(u_p_1),wires=q_1)
        elif 'RX' in item:
            pass
        elif 'CNOT' in item:
            temp=item.replace('CNOT ','')
            q_1=temp[0]
            q_2=temp[2]
            qml.CNOT(wires=[q_1, q_2])
    return qml.expval(qml.PauliZ(0)) #fix this
    import itertools
import numpy as np
from functools import partial
from pyquil import Program, api
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, sZ
from pyquil.gates import *
from scipy.optimize import minimize
#from forest_tools import *
import pennylane as qml
from pennylane import numpy as np
np.set_printoptions(precision=3, suppress=True)

import re

def create_circuit(beta, gamma,initial_state,exp_Hm,exp_Hc):
    circuit = Program()
    circuit += initial_state
    for i in range(p):
        for term_exp_Hc in exp_Hc:
            circuit += term_exp_Hc(-beta[i])
        for term_exp_Hm in exp_Hm:
            circuit += term_exp_Hm(-gamma[i])

    return circuit

# set p beforehand
p = 2

# dev = qml.device('default.qubit', wires=2)

# @qml.qnode(dev, interface='torch')
def QAOA_circ(parameters):# = np.random.uniform(0, np.pi*2, 2*p)):
    
    beta = parameters[:p]
    gamma = parameters [p:]
    
    def set_up_QAOA_in_pyquil(beta, gamma, p , n_qubits = 2, J = np.array([[0,1],[0,0]])):
                
        Hm = [PauliTerm("X", i, 1.0) for i in range(n_qubits)]
        
        Hc = []
        ####################Prepare the hamiltonian for measurement
        Hamilton=prepare_qaoa_hamiltonian(J,n_qubits)
        ###################
        initial_state = Program()
        for i in range(n_qubits):
            initial_state += H(i)      
        for i in range(n_qubits):
            for j in range(n_qubits):
                Hc.append(PauliTerm("Z", i, -J[i, j]) * PauliTerm("Z", j, 1.0))
        exp_Hm = []
        exp_Hc = []
        for term in Hm:
            exp_Hm.append(exponential_map(term))
        for term in Hc:
            exp_Hc.append(exponential_map(term)) 
        qaoa_circuit = create_circuit(beta, gamma,initial_state,exp_Hm,exp_Hc)
        
        return Hamilton,qaoa_circuit
    
    Hamilton,pyquil_circ=set_up_QAOA_in_pyquil(beta, gamma, p)
    pyquil_circ_list=str(pyquil_circ).split('\n')
    for item in pyquil_circ_list:
        u_p_1=None
        q_1=None
        q_2=None
        u_p_2=None
        u_p_3=None
        if 'H' in item: 
            q_1=item[item.find('H')+2] 
            qml.Hadamard(wires=q_1)
        elif 'RZ(' in item:
            temp=item.replace('RZ(','')
            u_p_1=temp[:temp.find(')')]
            q_1=temp[temp.find(')')+2]
            qml.RZ(float(u_p_1),wires=q_1)
        elif 'RX' in item:
            pass
        elif 'CNOT' in item:
            temp=item.replace('CNOT ','')
            q_1=temp[0]
            q_2=temp[2]
            qml.CNOT(wires=[q_1, q_2])
        wiress=[i for i in range(n_qubits)]
    return qml.expval.Hermitian(Hamilton,wires=wiress) 
