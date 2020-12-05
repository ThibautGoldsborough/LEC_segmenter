import matplotlib.pyplot as plt
import numpy as np


    
X = np.array(())
Y= np.array(())
U = np.array(())
V = np.array(())

for cell_index in list(CELL_DICTIONARY.keys()):
    startx=CELL_DICTIONARY[cell_index][0][0][0]
    starty=CELL_DICTIONARY[cell_index][0][0][1]
    X=np.append(X,startx)
    Y=np.append(Y,starty)
    
    endx=CELL_DICTIONARY[cell_index][len(CELL_DICTIONARY[cell_index])-1][0][0]
    endy=CELL_DICTIONARY[cell_index][len(CELL_DICTIONARY[cell_index])-1][0][1]
    vectX=startx-endx
    vectY=starty-endy
    
    U=np.append(U,vectX)
    V=np.append(V,vectY)
    
    

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V,units='xy' ,scale=0.2,headwidth=2)

plt.grid()

ax.set_aspect('equal')

plt.xlim(0,512)
plt.ylim(0,512)


plt.show()
