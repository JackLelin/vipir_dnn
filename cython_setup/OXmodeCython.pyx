import cython
import numpy as np

from libc.math cimport sqrt, pow, ceil, sin, cos, exp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
@cython.cpow(True)
def OXmode_cython(double[:] fps, double[:] fes, double dz, double df = 0.025 , double[:] freqs = None):
    # O-trace forward model
    cdef int i, j
    cdef double fcO = 0 # critical frequency in O-mode
    cdef double fe0 = 0 # electron gyro-frequency in MHz
    cdef int NY = fps.shape[0]
    for j in range(NY): 
        if fcO < fps[j]:
            fcO = fps[j]
            fe0 = fes[j]
    fvsO=np.arange(41.,ceil(40*fcO)+1, dtype = np.float64)/40.     # setting the frequencies to skim through
    vhsO=np.zeros_like(fvsO) #virtual height array initiated
    cdef double[:] fvsO_view = fvsO
    cdef double[:] vhsO_view = vhsO
    cdef double vh, n2b, n2t, fp, fv, fe, X, Y, prev_dh, prevn2b
    
    cdef int NX = fvsO_view.shape[0] 
    for i in range(NX):
    # for fv in fvsO: # for all ionosonde frequencies
        fv = fvsO_view[i]
        vh=0
        n2b=1
        prev_dh = 0
        prevn2b = 1
        for j in range(NY): # for all heights until reflection height
            fp = fps[j] 
            X=pow(fp/fv,2)
            n2t=1-X
            if n2t>0:
                prev_dh = dz*((1/sqrt(n2b)+1/sqrt(n2t))/2.) # keep track of latest dh
                prevn2b = n2b                                     # keep track of latest nb2
                vh=vh+prev_dh
                n2b=n2t
            else:
                # substract last dh and add last slab made of two slabs:
                vh=vh - prev_dh + 2.*(2*dz)*sqrt(prevn2b)/(prevn2b-n2t)
                break
        vhsO_view[i]= vh #virtual height array is built

    # X-trace forward model
    cdef double fcX=(sqrt(4*fcO*fcO+fe0+fe0)+fe0)/2        # critical frequency in X-mode
    cdef double fxb, fxt
    fvsX = np.arange(41.,ceil(40*fcX)+1, dtype = np.float64)/40.     # setting the frequencies to skim through
    vhsX = np.zeros_like(fvsX)
    cdef double[:] fvsX_view = fvsX
    cdef double[:] vhsX_view = vhsX
    NX = fvsX_view.shape[0] 
    for i in range(NX):
        fv = fvsX_view[i]
        vh=0
        n2b=1
        fxb=1
        prev_dh = 0
        prevn2b = 1
        for j in range(NY):
            fp = fps[j]
            fe = fes[j]
            X=pow(fp/fv,2)
            Y=fe/fv
            n2t=1-X*(1-X)/(1-X-pow(Y,2))
            fxt=n2t+X*pow((1-X)/(1-X-pow(Y,2)),2)+pow(X*Y/(1-X-pow(Y,2)),2)
            if n2t>0:
#                 prev_dh = dz/((sqrt(n2b)/fxb+sqrt(n2t)/fxt)/2.)
                prev_dh = dz*(fxb/sqrt(n2b)+fxt/sqrt(n2t))/2.     # keep track of latest dh
                prevn2b = n2b                                     # keep track of latest n2b
                vh=vh + prev_dh
                n2b=n2t
                fxb=fxt
            else:
                vh=vh - prev_dh + 2.*(2*dz)*sqrt(prevn2b)/(prevn2b-n2t)*(2-Y)/(1-Y)
                break
        vhsX_view[i] = vh #virtual height array is built

    return fvsO,vhsO,fvsX,vhsX