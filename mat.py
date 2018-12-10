# Copyright 2013 Philip N. Klein
from vec import Vec
import collections
#Test your Mat class over R and also over GF(2).  The following tests use only R.


# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

class RowAndVecLengthNotEqual(Error):
   """Raised when the length of the Vector is not equal to the length of rows in the Matrix"""
   pass

class ColLengthNotEqual(Error):
   """Raised when the length of cols in Matrix A is not equal to the length of cols in Matrix B"""
   pass

class RowAndColLengthNotEqual(Error):
   """Raised when the length of the Vector is not equal to the length of rows in the Matrix"""
   pass

def getitem(M, k):
    """
    Returns the value of entry k in M, where k is a 2-tuple
    >>> M = Mat(({1,3,5}, {'a'}), {(1,'a'):4, (5,'a'): 2})
    >>> M[1,'a']
    4
    >>> M[3,'a']
    0
    """
    try:
        return dict.__getitem__(M.f, k)
    except KeyError:
        return 0

def equal(A, B):
    """
    Returns True iff A is equal to B.

    Consider using brackets notation A[...] and B[...] in your procedure
    to access entries of the input matrices.  This avoids some sparsity bugs.

    >>> Mat(({'a','b'}, {'A','B'}), {('a','B'):0}) == Mat(({'a','b'}, {'A','B'}), {('b','B'):0})
    True
    >>> A = Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1})
    >>> B = Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1, ('b','B'):0})
    >>> C = Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1, ('b','B'):5})
    >>> A == B
    True
    >>> B == A
    True
    >>> A == C
    False
    >>> C == A
    False
    >>> A == Mat(({'a','b'}, {'A','B'}), {('a','B'):2, ('b','A'):1})
    True
    """

    if(len(A.D) == len(B.D)):
        M1 = A.copy()
        m1Rows = M1.D[0]
        m1Cols = M1.D[1]
        M2 = B.copy()
        m2Rows = M2.D[0]
        m2Cols = M2.D[1]
        for i in range(len(m1Rows)):
            for j in range(len(m1Cols)):
                if(M1[m1Rows[i],m1Cols[j]] != M2[m2Rows[i],m2Cols[j]]):
                    return False
        return True
    else:
        return False

def setitem(M, k, val):
    """
    Set entry k of Mat M to val, where k is a 2-tuple.
    >>> M = Mat(({'a','b','c'}, {5}), {('a', 5):3, ('b', 5):7})
    >>> M['b', 5] = 9
    >>> M['c', 5] = 13
    >>> M == Mat(({'a','b','c'}, {5}), {('a', 5):3, ('b', 5):9, ('c',5):13})
    True

    Make sure your operations work with bizarre and unordered keys.

    >>> N = Mat(({((),), 7}, {True, False}), {})
    >>> N[(7, False)] = 1
    >>> N[(((),), True)] = 2
    >>> N == Mat(({((),), 7}, {True, False}), {(7,False):1, (((),), True):2})
    True
    """
    if (True != (k[0] in M.D[0] and k[1] in M.D[1])):
        M.D[0].append(k[0])
        M.D[1].append(k[1])
    dict.__setitem__(M.f, k, val)

def add(A, B):
    """
    Return the sum of Mats A and B.

    Consider using brackets notation A[...] or B[...] in your procedure
    to access entries of the input matrices.  This avoids some sparsity bugs.

    >>> A1 = Mat(({3, 6}, {'x','y'}), {(3,'x'):-2, (6,'y'):3})
    >>> A2 = Mat(({3, 6}, {'x','y'}), {(3,'y'):4})
    >>> B = Mat(({3, 6}, {'x','y'}), {(3,'x'):-2, (3,'y'):4, (6,'y'):3})
    >>> A1 + A2 == B
    True
    >>> A2 + A1 == B
    True
    >>> A1 == Mat(({3, 6}, {'x','y'}), {(3,'x'):-2, (6,'y'):3})
    True
    >>> zero = Mat(({3,6}, {'x','y'}), {})
    >>> B + zero == B
    True
    >>> C1 = Mat(({1,3}, {2,4}), {(1,2):2, (3,4):3})
    >>> C2 = Mat(({1,3}, {2,4}), {(1,4):1, (1,2):4})
    >>> D = Mat(({1,3}, {2,4}), {(1,2):6, (1,4):1, (3,4):3})
    >>> C1 + C2 == D
    True
    """

    M1 = A.copy() if len(A.D)>=len(B.D) else B.copy()
    m1Rows = M1.D[0]
    m1Cols = M1.D[1]
    M2 = B.copy() if len(A.D)>=len(B.D) else A.copy()
    m2Rows = M2.D[0]
    m2Cols = M2.D[1]
    test = m2Rows
    test = m2Cols

    for x in range(len(m2Rows)):
        for y in range(len(m2Cols)):
            M1[m1Rows[x],m1Cols[y]] = M1[m1Rows[x],m1Cols[y]]+M2[m2Rows[x],m2Cols[y]]
    return M1

def scalar_mul(M, x):
    """
    Returns the result of scaling M by x.

    >>> M = Mat(({1,3,5}, {2,4}), {(1,2):4, (5,4):2, (3,4):3})
    >>> 0*M == Mat(({1, 3, 5}, {2, 4}), {})
    True
    >>> 1*M == M
    True
    >>> 0.25*M == Mat(({1,3,5}, {2,4}), {(1,2):1.0, (5,4):0.5, (3,4):0.75})
    True
    """
    R = M.copy()
    rows = M.D[0]
    cols = M.D[1]

    for a in range(len(rows)):
        for b in range(len(cols)):
            R[rows[a],cols[b]] = x*M[rows[a],cols[b]]

    return R

def transpose(M):
    """
    Returns the matrix that is the transpose of M.

    >>> M = Mat(({0,1}, {0,1}), {(0,1):3, (1,0):2, (1,1):4})
    >>> M.transpose() == Mat(({0,1}, {0,1}), {(0,1):2, (1,0):3, (1,1):4})
    True
    >>> M = Mat(({'x','y','z'}, {2,4}), {('x',4):3, ('x',2):2, ('y',4):4, ('z',4):5})
    >>> Mt = Mat(({2,4}, {'x','y','z'}), {(4,'x'):3, (2,'x'):2, (4,'y'):4, (4,'z'):5})
    >>> M.transpose() == Mt
    True
    """
    rows = M.D[0]
    cols = M.D[1]
    R = Mat((M.D[1], M.D[0]), {})

    for a in range(len(rows)):
        for b in range(len(cols)):
            rLen = len(rows) - 1
            cLen = len(cols) - 1
            R[cols[b], rows[a]] = M[rows[a], cols[b]]
    return R

def vector_matrix_mul(v, M):
    """
    returns the product of vector v and matrix M

    Consider using brackets notation v[...] in your procedure
    to access entries of the input vector.  This avoids some sparsity bugs.

    >>> v1 = Vec({1, 2, 3}, {1: 1, 2: 8})
    >>> M1 = Mat(({1, 2, 3}, {'a', 'b', 'c'}), {(1, 'b'): 2, (2, 'a'):-1, (3, 'a'): 1, (3, 'c'): 7})
    >>> v1*M1 == Vec({'a', 'b', 'c'},{'a': -8, 'b': 2, 'c': 0})
    True
    >>> v1 == Vec({1, 2, 3}, {1: 1, 2: 8})
    True
    >>> M1 == Mat(({1, 2, 3}, {'a', 'b', 'c'}), {(1, 'b'): 2, (2, 'a'):-1, (3, 'a'): 1, (3, 'c'): 7})
    True
    >>> v2 = Vec({'a','b'}, {})
    >>> M2 = Mat(({'a','b'}, {0, 2, 4, 6, 7}), {})
    >>> v2*M2 == Vec({0, 2, 4, 6, 7},{})
    True
    >>> v3 = Vec({'a','b'},{'a':1,'b':1})
    >>> M3 = Mat(({'a', 'b'}, {0, 1}), {('a', 1): 1, ('b', 1): 1, ('a', 0): 1, ('b', 0): 1})
    >>> v3*M3 == Vec({0, 1},{0: 2, 1: 2})
    True
    """
    r = v.copy()
    vecs = v.D
    rows = M.D[0]
    cols = M.D[1]
    print(len(vecs) != len(rows))
    try:
        if len(vecs) != len(rows):
            raise RowAndVecLengthNotEqual
        else:
            for a in range(len(rows)):
                sum = 0
                for b in range(len(cols)):
                    sum = sum + v[vecs[b]]*M[rows[a],cols[b]]
                r[vecs[a]] = sum

            return r
    except RowAndVecLengthNotEqual:
        return "Raised when the length of the Vector is not equal to the length of rows in the Matrix"

def matrix_vector_mul(M, v):
    """
    Returns the product of matrix M and vector v.

    Consider using brackets notation v[...] in your procedure
    to access entries of the input vector.  This avoids some sparsity bugs.

    >>> N1 = Mat(({1, 3, 5, 7}, {'a', 'b'}), {(1, 'a'): -1, (1, 'b'): 2, (3, 'a'): 1, (3, 'b'):4, (7, 'a'): 3, (5, 'b'):-1})
    >>> u1 = Vec({'a', 'b'}, {'a': 1, 'b': 2})
    >>> N1*u1 == Vec({1, 3, 5, 7},{1: 3, 3: 9, 5: -2, 7: 3})
    True
    >>> N1 == Mat(({1, 3, 5, 7}, {'a', 'b'}), {(1, 'a'): -1, (1, 'b'): 2, (3, 'a'): 1, (3, 'b'):4, (7, 'a'): 3, (5, 'b'):-1})
    True
    >>> u1 == Vec({'a', 'b'}, {'a': 1, 'b': 2})
    True
    >>> N2 = Mat(({('a', 'b'), ('c', 'd')}, {1, 2, 3, 5, 8}), {})
    >>> u2 = Vec({1, 2, 3, 5, 8}, {})
    >>> N2*u2 == Vec({('a', 'b'), ('c', 'd')},{})
    True
    >>> M3 = Mat(({0,1},{'a','b'}),{(0,'a'):1, (0,'b'):1, (1,'a'):1, (1,'b'):1})
    >>> v3 = Vec({'a','b'},{'a':1,'b':1})
    >>> M3*v3 == Vec({0, 1},{0: 2, 1: 2})
    True
    """
    vecs = v.D
    rows = M.D[0]
    cols = M.D[1]
    r = Vec(M.D[0],{})
    try:
        if len(vecs) != len(rows):
            raise RowAndVecLengthNotEqual
        else:
            for a in range(len(rows)):
                sum = 0
                for b in range(len(cols)):
                    sum = sum + v[vecs[b]]*M[rows[a],cols[b]]
                r[rows[a]] = sum

            return r
    except RowAndVecLengthNotEqual:
        return "Raised when the length of the Vector is not equal to the length of rows in the Matrix"

def matrix_matrix_mul(A, B):
    """
    Returns the result of the matrix-matrix multiplication, A*B.

    Consider using brackets notation A[...] and B[...] in your procedure
    to access entries of the input matrices.  This avoids some sparsity bugs.

    >>> A = Mat(({0,1,2}, {0,1,2}), {(1,1):4, (0,0):0, (1,2):1, (1,0):5, (0,1):3, (0,2):2})
    >>> B = Mat(({0,1,2}, {0,1,2}), {(1,0):5, (2,1):3, (1,1):2, (2,0):0, (0,0):1, (0,1):4})
    >>> A*B == Mat(({0,1,2}, {0,1,2}), {(0,0):15, (0,1):12, (1,0):25, (1,1):31})
    True
    >>> C = Mat(({0,1,2}, {'a','b'}), {(0,'a'):4, (0,'b'):-3, (1,'a'):1, (2,'a'):1, (2,'b'):-2})
    >>> D = Mat(({'a','b'}, {'x','y'}), {('a','x'):3, ('a','y'):-2, ('b','x'):4, ('b','y'):-1})
    >>> C*D == Mat(({0,1,2}, {'x','y'}), {(0,'y'):-5, (1,'x'):3, (1,'y'):-2, (2,'x'):-5})
    True
    >>> M = Mat(({0, 1}, {'a', 'c', 'b'}), {})
    >>> N = Mat(({'a', 'c', 'b'}, {(1, 1), (2, 2)}), {})
    >>> M*N == Mat(({0,1}, {(1,1), (2,2)}), {})
    True
    >>> E = Mat(({'a','b'},{'A','B'}), {('a','A'):1,('a','B'):2,('b','A'):3,('b','B'):4})
    >>> F = Mat(({'A','B'},{'c','d'}),{('A','d'):5})
    >>> E*F == Mat(({'a', 'b'}, {'d', 'c'}), {('b', 'd'): 15, ('a', 'd'): 5})
    True
    >>> F.transpose()*E.transpose() == Mat(({'d', 'c'}, {'a', 'b'}), {('d', 'b'): 15, ('d', 'a'): 5})
    True
    """
    M1 = A.copy() if len(A.D)>=len(B.D) else B.copy()
    m1Rows = M1.D[0]
    m1Cols = M1.D[1]
    M2 = B.copy() if len(A.D)>=len(B.D) else A.copy()
    m2Rows = M2.D[0]
    m2Cols = M2.D[1]

    try:
        if len(m1Cols) != len(m2Rows):
            raise RowAndColLengthNotEqual
        else:
            R = Mat((M1.D[1],M2.D[0]),{})
            rRows = R.D[0]
            rCols = R.D[1]
            for i in range(len(rRows)):
                for j in range(len(rCols)):
                    sum = 0
                    for n in range(len(rCols)):
                        sum = sum + M1[m1Rows[i],m1Cols[n]]*M2[m2Rows[n],m2Cols[j]]
                    R[rRows[i],rCols[j]] = sum
            return R
    except RowAndColLengthNotEqual:
        return "Raised when the length of Matrix A's rows are not equal to the length of cols of Matrix B"

def addRows(A, B):
    M1 = A.copy() if len(A.D)>=len(B.D) else B.copy()
    m1Rows = M1.D[0]
    m1Cols = M1.D[1]
    M2 = B.copy() if len(A.D)>=len(B.D) else A.copy()
    m2Rows = M2.D[0]
    m2Cols = M2.D[1]

    try:
        if len(m1Cols) != len(m2Cols):
            raise ColLengthNotEqual
        else:
            e = m1Rows[-1]
            secondRow = set([x+e+1 for x in m2Rows])
            R = Mat((M1.D[0].union(secondRow),M1.D[1]),{})
            rRows = R.D[0]
            rCols = R.D[1]
            for i in range(len(m1Rows)):
                for j in range(len(rCols)):
                    R[rRows[i],rCols[j]] = M1[m1Rows[i],m1Cols[j]]
            for i in range(len(m2Rows)):
                for j in range(len(rCols)):
                    R[rRows[i+(len(m1Rows))],rCols[j]] = M2[m2Rows[i],m2Cols[j]]
            return R
    except ColLengthNotEqual:
        return ("Raised when the length of cols in Matrix ",A.__name__," is not equal to the length of cols in Matrix ",B.__name__)

def appendColVec(self, v):
    rows = self.D[0]
    cols = self.D[1]
    pos = v.D
    for i in range(len(rows)):
        self[(rows[i],('y4','x4'))] = v[pos[i]]

def getRowVec(M,k):
        rows = M.D[0]
        cols = M.D[1]
        r = Vec(M.D[1],{})
        for i in range(len(cols)):
            r[cols[i]] = M[rows[k],cols[i]]
        return r

def prettyprint(M):
    rows=None
    cols=None
    "string representation for print()"
    if rows == None: rows = sorted(M.D[0], key=repr)
    if cols == None: cols = sorted(M.D[1], key=repr)
    separator = ' | '
    numdec = 3
    pre = 1+max([len(str(r)) for r in rows])

    colw = {col:(1+max([len(str(col))] + [len('{0:.{1}G}'.format(M[row,col],numdec)) if isinstance(M[row,col], int) or isinstance(M[row,col], float) else len(str(M[row,col])) for row in rows])) for col in cols}

    s1 = ' '*(1+ pre + len(separator))
    s2 = ''.join(['{0:>{1}}'.format(str(c),colw[c]) for c in cols])
    s3 = ' '*(pre+len(separator)) + '-'*(sum(colw.values()) + 1)

    s4 = ''.join(['{0:>{1}} {2}'.format(str(r), pre,separator)+''.join(['{0:>{1}.{2}G}'.format(M[r,c],colw[c],numdec) if isinstance(M[r,c], int) or isinstance(M[r,c], float) else '{0:>{1}}'.format(M[r,c], colw[c]) for c in cols])+'\n' for r in rows])

    return '\n' + s1 + s2 + '\n' + s3 + '\n' + s4

################################################################################

class Mat:
    def __init__(self, labels, function):
        assert isinstance(labels, tuple)
        assert isinstance(labels[0], list) and isinstance(labels[1], list)
        assert isinstance(function, dict)
        self.D = labels
        self.f = function

    appendColVec = appendColVec
    prettyprint = prettyprint
    __getitem__ = getitem
    __setitem__ = setitem
    transpose = transpose
    addRows = addRows
    getRowVec = getRowVec

    def __neg__(self):
        return (-1)*self

    def __mul__(self,other):
        if Mat == type(other):
            return matrix_matrix_mul(self,other)
        elif Vec == type(other):
            return matrix_vector_mul(self,other)
        else:
            return scalar_mul(self,other)
            #this will only be used if other is scalar (or not-supported). mat and vec both have __mul__ implemented

    def __rmul__(self, other):
        if Vec == type(other):
            return vector_matrix_mul(other, self)
        else:  # Assume scalar
            return scalar_mul(self, other)

    __add__ = add

    def __radd__(self, other):
        "Hack to allow sum(...) to work with matrices"
        if other == 0:
            return self

    def __sub__(a,b):
        return a+(-b)

    __eq__ = equal

    def copy(self):
        return Mat(self.D, self.f.copy())

    def pp(self, rows, cols):
        print(self.__str__(rows, cols))

    def __str__(self):
        "evaluatable representation"
        return "Mat(" + str(self.D) +", " + str(self.f) + ")"

    def __iter__(self):
        raise TypeError('%r object is not iterable' % self.__class__.__name__)
