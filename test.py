import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
import glob
img=[]
a=sp.misc.imread("rohith.jpg")
dim1=100
dim2=100
a=sp.misc.imresize(a,[dim1,dim2])
a=a.tolist()
for i in range(0,dim1):
	d=[]
	for j in range(0,dim2):
		p=0.2126*a[i][j][0]+0.7152*a[i][j][1]+0.0722*a[i][j][2]
		d.append(p)
	img.append(d)
#print img
plt.imshow(img)
plt.show()
w=np.matrix(u)