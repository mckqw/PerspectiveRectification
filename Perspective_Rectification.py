import vecutil
import matutil
from mat import Mat
from vec import Vec
import matutil
import itertools
from sympy import *

C = ['x1','x2','x3']

R = ['y1','y2','y3']

c1 = Vec(['y1','y2','y3'],{'y1':1,'y2':0,'y3':1})
c2 = Vec(['y1','y2','y3'],{'y1':0,'y2':1,'y3':1})
c3 = Vec(['y1','y2','y3'],{'y1':0,'y2':0,'y3':1})
c4 = Vec(['y1','y2','y3'],{'y1':1,'y2':1,'y3':1})
yCords = [c1,c2,c3,c4]

def pixel(x): return(x[0], x[1])

def move2board(y):
    yPos = y.D
    z = Vec(['y1','y2','y3'],{'y1':y[yPos[0]]/y[yPos[2]],'y2':y[yPos[1]]/y[yPos[2]],'y3':y[yPos[2]]/y[yPos[2]]})
    return z

def cartesian_product_loop(tup1, tup2):
    res = ()
    for t1 in tup1:
        for t2 in tup2:
            res += ((t1, t2), (t2, t1))
    return res

D = list()
for i in itertools.product(['y1','y2','y3'],['x1','x2','x3']):
         D.append(i)

#h = Vec(D, {('y1','x1'): H[('y1','x1')], ('y1','x2'):H[('y1','x2')], ('y1','x3'):H[('y1','x3')], ('y2','x1'): H[('y2','x1')], ('y2','x2'):H[('y2','x2')], ('y2','x3'):H[('y2','x3')], ('y3','x1'): H[('y3','x1')], ('y3','x2'):H[('y3','x2')], ('y3','x3'):H[('y3','x3')]})
h = Vec(D, {})

def make_equations(x1, x2, w1, w2):
    equ = list(())

    """
    u = Vec(D, {('y3','x1'): w1 * x1 * h[('y3','x1')], ('y3','x2'): w1 * x2 * h[('y3','x2')], ('y3','x3'): w1 * h[('y3','x3')], ('y1','x1'): -x1 * h[('y1','x1')], ('y1','x2'): -x2 * h[('y1','x2')], ('y1','x3'): -1 * h[('y1','x3')]})
    v = Vec(D, {('y3','x1'): w2 * x1 * h[('y3','x1')], ('y3','x2'): w2 * x2 * h[('y3','x2')], ('y3','x3'): w2 * h[('y3','x3')], ('y2','x1'): -x1 * h[('y2','x1')], ('y2','x2'): -x2 * h[('y2','x2')], ('y2','x3'): -1 * h[('y2','x3')]})

    u = Vec(D, {('y1','x1'): x1, ('y1','x2'):x2, ('y1','x3'):1})
    v = Vec(D, {('y3','x1'): x1, ('y3','x2'): x2, ('y3','x3'): 1, ('y2','x1'): -x1, ('y2','x2'): -x2, ('y2','x3'): -1})
    """
    u = Vec(D, {('y3','x1'): (w1 * x1), ('y3','x2'): (w1 * x2), ('y3','x3'): w1, ('y1','x1'): -x1, ('y1','x2'): -x2, ('y1','x3'): -1})
    v = Vec(D, {('y3','x1'): (w2 * x1), ('y3','x2'): (w2 * x2), ('y3','x3'): w2, ('y2','x1'): -x1, ('y2','x2'): -x2, ('y2','x3'): -1})
    equ.append(u)
    equ.append(v)
    return equ

w = Vec(D, {('y1','x1'): 1})

b = Vec([0,1,2,3,4,5,6,7,8],{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1})

def make_nine_equations(corners):
    veclist = list(())
    for n in range(len(corners)):
        x1 = corners[n][0]
        x2 = corners[n][1]
        y1 = yCords[n]['y1']
        y2 = yCords[n]['y2']
        y3 = yCords[n]['y3']
        hold = make_equations(x1,x2,y1/y3,y2/y3)
        veclist.append(hold[0])
        veclist.append(hold[1])
    veclist.append(w)
    return veclist

veclist = make_nine_equations([(358,36), (329,597), (592,157), (580,483)])

L = matutil.rowdict2mat(veclist)
L.appendColVec(b)
print(L.prettyprint())

# Perform Gaussian Elimination
rows = L.D[0]
cols = L.D[1]

hold = []
for j in range(0, len(cols)):
    hold.append(L[rows[0],cols[j]])
M = Matrix([hold])

for i in range(1, len(rows)-1):
    hold = []
    for j in range(0, len(cols)):
        h = L[rows[i],cols[j]]
        hold.append(h)
    M = M.row_insert(i, Matrix([hold]))
M = M.rref()
M = M[0]
H = Mat((C,R),{})
rows = H.D[0]
cols = H.D[1]

n=0
for i in range(0, 3):
    for j in range(0, 3):
        print(M.row(n)[9])
        H[rows[j],cols[i]] = M.row(n)[9]
        n+=1

print(H.prettyprint())
testV = Vec(['x1','x2','x3'], {'x1':592, 'x2':157, 'x3':1})
print(testV * H)


#
