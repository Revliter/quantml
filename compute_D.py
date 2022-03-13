import numpy as np
'''
compute B_bar and D
e_a: e^(-ρ_a*τ)
e_b: e^(-ρ_b*τ)
lamda: Λ
Q_a: Q^a
Q_b: Q^b
'''
def compute_A(m, e_a: np.matrix, e_b: np.matrix)-> np.matrix:
    A = np.block([
        [np.eye(m+1), np.zeros(shape=(m+1,m)), np.zeros(shape=(m+1,m))],
        [np.zeros(shape=(m,m+1)), e_a, np.zeros(shape=(m,m))],
        [np.zeros(shape=(m,m+1)), np.zeros(shape=(m,m)), e_b]
    ])
    return A

def compute_B(m, e_a: np.matrix, e_b: np.matrix, Q_a: np.matrix, Q_b: np.matrix, lamda: np.matrix)-> np.matrix:
    delta = np.block([
        [np.eye(m)],
        [-np.eye(m)]
    ])
    delta_a = np.block([
        [np.eye(m)],
        [np.zeros(shape=(m,m))]
    ])
    delta_b = np.block([
        [np.zeros(shape=(m,m))],
        [-np.eye(m)]
    ])
    kappa_a = 2*np.dot(Q_a, np.transpose(delta_a))-np.dot(lamda, np.transpose(delta))
    kappa_b = 2*np.dot(Q_b, np.transpose(delta_b))+np.dot(lamda, np.transpose(delta))
    B = np.block([
        [np.zeros(shape=(1,2*m))],
        [-np.transpose(delta)],
        [np.dot(e_a, kappa_a)],
        [np.dot(e_b, kappa_b)]
    ])
    return B

def compute_Bbar(A: np.matrix, B: np.matrix, N: np.matrix, m)-> np.matrix:
    B_bar = np.random.random(size = ((N+1)*(3*m+1), (N+1)*(2*m)))
    Row = B_bar.shape[0]
    Col = B_bar.shape[1]
    row_A = A.shape[0]
    col_A = A.shape[1]
    row_B = B.shape[0]
    col_B = B.shape[1]
    
    for i in range(0,N+1): # i row block
        if i == 0:
            B_bar[0:row_B,:] = np.zeros(shape=(row_B, Col))
        else:
            posy = i*row_B
            for j in range(0,N+1):
                posx = j*col_B
                if i-1-j >= 0:
                    tmp = np.eye(row_A, col_A)
                    for k in range(0,i-1-j):
                        tmp = np.dot(tmp,A)
                    B_bar[posy:posy + row_B,posx:posx + col_B] = np.dot(tmp,B)
                else:
                    B_bar[posy:posy + row_B,posx:posx + col_B] = np.zeros(shape = (row_B,col_B))
    return B_bar

def compute_D(B_bar: np.matrix, N_bar: np.matrix, Q_bar: np.matrix)->np.matrix :
    D = 0.5 * (np.dot(np.transpose(B_bar),N_bar) + np.dot(np.transpose(N_bar),B_bar)) + Q_bar
    return D
