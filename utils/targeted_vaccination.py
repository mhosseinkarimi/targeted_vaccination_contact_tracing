import numpy as np
import matplotlib.pyplot as plt


def calculate_herd_effect(i0, beta, gamma, vac_k, vac_f,
                          max_iter=50, eps=0.001):
    # initialize SIR
    i_1 = i0
    s_1 = 1 - i0
    r_1 = 0
    
    for k in range(max_iter):
        # calculating the derivatives
        ds = - beta * s_1 * i_1
        di = (beta * s_1 * i_1) - (gamma * i_1)
        dr = gamma * i_1
        
        # Vaccination
        if k == vac_k:
            if vac_f <= s_1:
                s_1 -= vac_f
                r_1 += vac_f
            else: 
                H = 0
                return H
        
        # updating the values
        s = s_1 + ds if s_1+ds > 0 else 0
        i= i_1 + di if i_1+di > 0 else 0
        r = r_1 + dr if r_1+dr > 0 else 0
        
        # convergence check
        if np.abs(s - s_1) < eps and k > vac_k:
            H = s
            return H

        s_1 = s
        i_1 = i
        r_1 = r
        
    return s


def calculate_obj_fcn(N_vec, f_vec, i0_vec, 
                  tau, beta, gamma, max_iter=50):
    assert len(N_vec) == len(f_vec)
    
    H = np.zeros((len(N_vec,)))
    for k in range(len(N_vec)):
        H[k] = calculate_herd_effect(i0_vec[k], beta, gamma, tau, f_vec[k])
    obj_fcn = np.sum(N_vec * (f_vec + H))
    return obj_fcn

def optimize_obj_fcn(N_vec, V, i0_vec, tau, beta, gamma):
    if len(N_vec) == 2:
        f1 = np.arange(0, V/N_vec[0], 0.01)
        f2 = V/N_vec[1] - N_vec[0]/N_vec[1] * f1;
        # print(f1, f2)

        F = np.zeros((len(f1,)))
        
        for k in range(len(f1)):
            F[k] = calculate_obj_fcn(N_vec, [f1[k], f2[k]], i0_vec, tau, beta, gamma)
        F_max = np.max(F)
        arg_opt = np.argmax(F)
        f_opt = np.array([f1[arg_opt], f2[arg_opt]])
        plt.plot(f1, F)
        plt.savefig("tmp.png")
        return F_max, f_opt
        

if __name__ == "__main__":
    H = calculate_herd_effect(0.01, 0.3, 0.1, 10, 0.3)
    print(calculate_obj_fcn([1, 1], [0.1, 0.2], [0.01, 0.01], 10, 0.5, 0.1))
    print(optimize_obj_fcn([10, 12], 8, [0.01, 0.01], 15, 0.2, 0.1))