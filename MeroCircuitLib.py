import numpy as np

# This class represents the network data. R is the matrix of resistivities, such that R_{ij}
# Is the resistivity between the vertex i and the vertex j. Also, the voltage is specified on
# certain nodes (in-nodes), denoted as voltage_in (this is expected to be a list or an array).
# We inject external current on some node (inj_cur).
# Also, we specify the indices of in-nodes, in_nodes, and out-nodes, out_nodes 

class R_Network:
    def __init__(self, R, voltage_in, in_nodes, out_nodes):
        self.R = R
        self.voltage_in = np.array(voltage_in)
        self.in_nodes = np.array(in_nodes)
        self.out_nodes = np.array(out_nodes)

        
    def calculate_Laplacian(self):
        
        # The conductance matrix
        self.S = np.where(self.R != 0, 1.0 / self.R, 0.0)

        # The discrete laplacian matrix
        S_row_sums = np.sum(self.S, axis=1)
        L_disord = -self.S + np.diag(S_row_sums)

        # Next we bring the matrix to the block form in the following order: first the in-nodes, 
        # then the out-nodes, then the internal ones.
        all_nodes = np.arange(len(L_disord))
        inout_nodes = np.concatenate([self.in_nodes, self.out_nodes])
        int_nodes = np.setdiff1d(all_nodes, inout_nodes, assume_unique=True)
        self.new_order = np.concatenate([self.in_nodes, self.out_nodes, int_nodes])
        
        return L_disord[np.ix_(self.new_order, self.new_order)]    

    # This method executes the main taks of this cllass% computes the output voltage
    
    def calculate_voltage_out(self):
        
        L = self.calculate_Laplacian()
        
        n_in = len(self.in_nodes) # the number of the intput nodes
        n_out = len(self.out_nodes) # and of the output nodes
        
        L_UK = L[n_in:, :n_in] # the blocks of the Laplacian needed for the computation
        L_UU = L[n_in:, n_in:]

        V_K = np.array(self.voltage_in) # The known potentials

        # Solving the system L_UU * V_U = I_U - L_UK * V_K
        rhs = - L_UK @ V_K
        V_U = np.linalg.solve(L_UU, rhs)

        V_full = np.concatenate([V_K, V_U])
        
        # Creating a dictionary of potentials
        all_potentials = dict(zip(self.new_order, V_full))
        v_out_results = V_full[n_in : n_in + n_out]

        # Возвращаем v_out, словарь И полный вектор V_U для метода токов
        return v_out_results, all_potentials, V_U

    # The currents on the input nodes are not externally imposed, and in fact can be computed (optional method)
    
    def compute_current_in(self):
        
        L = self.calculate_Laplacian()

        n_in = len(self.in_nodes)
        
        L_KK = L[:n_in, :n_in]
        L_KU = L[:n_in, n_in:]

        V_K = np.array(self.voltage_in)
        _, _, V_U = self.calculate_voltage_out()
        
        current_in = L_KK@V_K + L_KU@V_U

        return current_in


# This class represents the network data for a non-linear network with (linear) capacitents.
# R is the matrix of resistivities, and C is the matrix of capacities. As before, the voltage is specified on
# certain nodes (in-nodes), denoted as voltage_in (this is expected to be a list or an array).
# We inject external current on some node (inj_cur).
# Also, we specify the indices of in-nodes, in_nodes, and out-nodes, out_nodes 
# While for applications it is interesting to consider the situation when initially the capasitors are not charged,
# here we will consider a slightly more general setup,  and introduce a matrix of initial  charges Q0.

class R_Network:
    def __init__(self, R, C, Q0, voltage_in, in_nodes, out_nodes):
        self.R = R
        self.C = C,
        self.Q0 = Q0,
        self.voltage_in = np.array(voltage_in)
        self.in_nodes = np.array(in_nodes)
        self.out_nodes = np.array(out_nodes)