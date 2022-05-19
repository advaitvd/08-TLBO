from ModalAnalysis import ModalAnalysis as ma
from TLBO import TLBO as tlbo_algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(test_num):
    file_name = '2D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    arrested_dofs=np.array([0,1,42,43])
    aa = ma(elements, nodes, dimension,arrested_dofs=arrested_dofs)
    M=aa.assembleMass()

    x_exp=np.zeros(len(elements))
    x_exp[5]=0.35
    x_exp[23]=0.20
    x_exp[15]=0.4   #localized
    x_exp[10]=0.24

    K=aa.assembleStiffness(x_exp)
    w_exp, v_exp=aa.solve_eig(K,aa.M)
    
    num_modes=10

    w_exp=w_exp[:num_modes]
    v_exp=v_exp[:,:num_modes]
    F_exp=np.sum(v_exp*v_exp,axis=0)/(w_exp*w_exp)
    # print("w_exp",w_exp)

    def objective_function(x):
        K=aa.assembleStiffness(x)
        w, v = aa.solve_eig(K, aa.M)
        w=w[:num_modes]
        v=v[:,:num_modes]
        # print(w.shape,v.shape)
        # print('w',w)
        
        MAC=(np.sum((v*v_exp),axis=0)**2)/(np.sum(v*v,axis=0)*np.sum(v_exp*v_exp,axis=0))
        
        F=np.sum(v*v,axis=0)/(w*w)
        MACF=(np.sum(F*F_exp)**2)/(np.sum(F*F)*np.sum(F_exp*F_exp))

        MDLAC=(np.abs(w-w_exp)/w_exp)**2

        # print('MAC, MDLAC',MAC, MDLAC)

        cost = np.sum(1-MAC)+np.sum(MDLAC)+np.sum(1-MACF)
        return cost

    optimizer = tlbo_algorithm(cost_fn=objective_function,n=len(elements),population_size=50,ub=0.99,lb=0,maxIter=200,verbose=True)
    
    best_solution,cost,logs=optimizer.run_optimizer(return_logs=True)
    # print(best_solution)
    # print(cost)
    plt.yscale("log")
    plt.plot(np.array(logs),'r-')
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title('TLBO Convergence')
    # plt.show()
    plt.savefig(f'./convergence_plots/convergence_{test_num+1}.png')
    plt.cla()

    return (best_solution,cost)



if __name__=='__main__':
    sols=[]
    costs=[]
    N=10
    for i in range(N):
        print(f'Run {i+1}/{N}')
        best_sol,cost=main(i)
        print("*"*80)
        sols.append(best_sol)
        costs.append(cost)
    
    sols=np.stack(sols)

    np.savetxt("best_sols.csv",sols,delimiter=',')