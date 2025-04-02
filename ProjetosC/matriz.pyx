cdef extern from 'matriz.sh'
    int *criaMatriz(int, int)

cdef class PyMatriz:
    cdef int *ptr
    cdef int n
    cdef int m

def __cinit__(self, int n, int n):
    self.ptr = criaMatriz(n, m)
    self.n = n
    self.m = m

def exibir(self):
    exibirMatriz(sefl.ptr, self.n self.m)