# %%
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import pylops
import matplotlib.pyplot as plt
from pylops.optimization.sparsity import fista
from pylops import MatrixMult
from scipy.fftpack import dct, idct

# %%
def DGI(patterns,image):
    x_flattened = image.flatten()
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    buckets = A @ x_flattened

    S = np.sum(patterns, axis=(1, 2))

    # Normalize using S
    B_norm = buckets / S 

    # Calc mean of the normalized buckets
    B_norm_avg = np.mean(B_norm)

    # Get I avg (light source)
    I_avg = np.mean(patterns, axis=0)

    # Now calc O(x,y) using the normalized measurements
    O = np.mean((B_norm[:, None, None] - B_norm_avg) * (patterns - I_avg), axis=0)

    #normalize 
    O = (O - O.min()) / (O.max() - O.min())

    return O

def PseudoInverse(patterns,image):
    x_flattened = image.flatten()
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    buckets = A @ x_flattened
    pseudo_inverse=np.linalg.pinv(A)@buckets
    final_recon=pseudo_inverse.reshape(256,256)
    final_recon = (final_recon - final_recon.min()) / (final_recon.max() - final_recon.min())
    return final_recon

def FISTA(patterns,image,eps=0.2):
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    x_flattened= image.flatten().astype(np.float32)
    ASparse = dct(A, norm='ortho')
    OpSparse = MatrixMult(ASparse @ np.identity(ASparse.shape[1]))

    buckets=A@x_flattened


    #do the fista function
    alpha_rec, niter_eff, cost = fista(OpSparse, y=buckets, niter=200, eps=eps, show=False)

    #reconstruct the image
    x_reconstruction = dct(alpha_rec, norm='ortho')
    final_recon = x_reconstruction.reshape(256,256)
    return final_recon

def DGI_Recon(patterns,buckets):

    S = np.sum(patterns, axis=(1, 2))

    # Normalize using S
    B_norm = buckets / S 

    # Calc mean of the normalized buckets
    B_norm_avg = np.mean(B_norm)

    # Get I avg (light source)
    I_avg = np.mean(patterns, axis=0)

    # Now calc O(x,y) using the normalized measurements
    O = np.mean((B_norm[:, None, None] - B_norm_avg) * (patterns - I_avg), axis=0)

    #normalize 
    O = (O - O.min()) / (O.max() - O.min())

    return O

def FISTA_Recon(patterns,buckets,eps=0.2):
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    ASparse = dct(A, norm='ortho')
    OpSparse = MatrixMult(ASparse @ np.identity(ASparse.shape[1]))

    #do the fista function
    alpha_rec, niter_eff, cost = fista(OpSparse, y=buckets, niter=200, eps=eps, show=False)

    #reconstruct the image
    x_reconstruction = dct(alpha_rec, norm='ortho')
    final_recon = x_reconstruction.reshape(256,256)
    return final_recon

def PseudoInverse_Recon(patterns,buckets):
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    pseudo_inverse=np.linalg.pinv(A)@buckets
    final_recon=pseudo_inverse.reshape(256,256)
    final_recon = (final_recon - final_recon.min()) / (final_recon.max() - final_recon.min())
    return final_recon

def DGI_SNR(patterns,image,SNR):
    x_flattened = image.flatten()
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    buckets = A @ x_flattened

    std = np.sqrt(np.mean(buckets**2) / (10 ** (SNR/10)))
    noise = np.random.normal(0, std, size=buckets.shape)
    buckets=buckets+noise

    S = np.sum(patterns, axis=(1, 2))

    # Normalize using S
    B_norm = buckets / S 

    # Calc mean of the normalized buckets
    B_norm_avg = np.mean(B_norm)

    # Get I avg (light source)
    I_avg = np.mean(patterns, axis=0)

    # Now calc O(x,y) using the normalized measurements
    O = np.mean((B_norm[:, None, None] - B_norm_avg) * (patterns - I_avg), axis=0)

    #normalize 
    O = (O - O.min()) / (O.max() - O.min())

    return O

def PseudoInverse_SNR(patterns,image,SNR):
    x_flattened = image.flatten()
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    buckets = A @ x_flattened

    std = np.sqrt(np.mean(buckets**2) / (10 ** (SNR/10)))
    noise = np.random.normal(0, std, size=buckets.shape)
    buckets=buckets+noise

    pseudo_inverse=np.linalg.pinv(A)@buckets
    final_recon=pseudo_inverse.reshape(256,256)
    final_recon = (final_recon - final_recon.min()) / (final_recon.max() - final_recon.min())
    return final_recon

def FISTA_SNR(patterns,image,SNR,eps=0.2):
    A=patterns.reshape(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    x_flattened= image.flatten().astype(np.float32)
    ASparse = dct(A, norm='ortho')
    OpSparse = MatrixMult(ASparse @ np.identity(ASparse.shape[1]))
    buckets=A@x_flattened

    std = np.sqrt(np.mean(buckets**2) / (10 ** (SNR/10)))
    noise = np.random.normal(0, std, size=buckets.shape)
    buckets=buckets+noise

    #do the fista function
    alpha_rec, niter_eff, cost = fista(OpSparse, y=buckets, niter=200, eps=eps, show=False)

    #reconstruct the image
    x_reconstruction = dct(alpha_rec, norm='ortho')

    final_recon = x_reconstruction.reshape(256,256)
    return final_recon