import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
import glob
import math
from sklearn import svm
p=glob.glob("barrel/*.jpg")
l=len(p)
u=[]
for i in range(0,l):
	a=sp.misc.imread(p[i])
	a=sp.misc.imresize(a,[50,50])
	a=a.tolist()
	g=[]
#	print a
	for j in range(1,len(a)):
		for k in range(1,len(a[j])):
			try:
				g.append(0.2126*a[j][k][0]+0.7152*a[j][k][1]+0.0722*a[j][k][2])
			except:
				continue
	u.append(g)


p=glob.glob("buddha/*.jpg")
l=len(p)
z=[]
for i in range(0,l):
	a=sp.misc.imread(p[i])
	a=sp.misc.imresize(a,[50,50])
	a=a.tolist()
	g=[]
#	print a
	for j in range(1,len(a)):
		for k in range(1,len(a[j])):
			try:
				g.append(0.2126*a[j][k][0]+0.7152*a[j][k][1]+0.0722*a[j][k][2])
			except:
				continue
	z.append(g)
u=u+z




p=glob.glob("elephant/*.jpg")
l=len(p)
z=[]
for i in range(0,l):
	a=sp.misc.imread(p[i])
	a=sp.misc.imresize(a,[50,50])
	a=a.tolist()
	g=[]
#	print a
	for j in range(1,len(a)):
		for k in range(1,len(a[j])):
			try:
				g.append(0.2126*a[j][k][0]+0.7152*a[j][k][1]+0.0722*a[j][k][2])
			except:
				continue
	z.append(g)
u=u+z


p=glob.glob("ferry/*.jpg")
l=len(p)
z=[]
for i in range(0,l):
	a=sp.misc.imread(p[i])
	a=sp.misc.imresize(a,[50,50])
	a=a.tolist()
	g=[]
#	print a
	for j in range(1,len(a)):
		for k in range(1,len(a[j])):
			try:
				g.append(0.2126*a[j][k][0]+0.7152*a[j][k][1]+0.0722*a[j][k][2])
			except:
				continue
	z.append(g)
u=u+z

p=glob.glob("panda/*.jpg")
l=len(p)
z=[]
for i in range(0,l):
	a=sp.misc.imread(p[i])
	a=sp.misc.imresize(a,[50,50])
	a=a.tolist()
	g=[]
#	print a
	for j in range(1,len(a)):
		for k in range(1,len(a[j])):
			try:
				g.append(0.2126*a[j][k][0]+0.7152*a[j][k][1]+0.0722*a[j][k][2])
			except:
				continue
	z.append(g)
u=u+z



count=0
while count<30:
	count=count+1
	h=u[count]
	e=[]
	for i in range(0,50):
		e.append(h[i*49:(i+1)*49])
	e=e[0:48]
	e=np.asarray(e)
	try:
		plt.gray()
		plt.imshow(e)
		plt.show()
	except:
		continue

# collecting data from images done
l=len(u)
r=len(u[0])
mean=[]
for i in range(0,len(u[0])):
	s=0
	for j in range(0,len(u)):
		try:
			s=s+u[j][i]
		except:
			continue
	s=s/l
	mean.append(s)
for i in range(0,len(u[0])):
	for j in range(0,len(u)):
		try:
			u[j][i]=u[j][i]-mean[u[i]]
		except:
			continue

# u contains centralised data
w=np.array(u[0])
l=len(u)
for i in range(1,l):
	try:
		w=np.vstack([w,np.array(u[i])])
	except:
		continue
wt=w.T 
x=np.dot(w,wt)
x=x/x.shape[0]
     # x is covarianve matrix
result=np.linalg.eig(x)
eigvector=result[1]
k=150
# we have eigvector.shape
eigvector=eigvector[0:k] 
eigvector=eigvector.T
l=len(eigvector)
#print l
#for i in range(0,l):
#	eigvector[i]=eigvector[i]/np.linalg.norm(eigvector[u[i])
#print wt.shape,eigvector.shape
u_re=np.dot(wt,eigvector)
u_re=np.dot(u_re,eigvector.T)
u_re=u_re.T
w_t=u_re[0]
l=len(u_re)
for i in range(1,l):
	try:
		w_t=np.vstack([w,np.array(u_re[i])])
	except:
		continue


for i in range(0,len(u_re[0])):
	for j in range(0,len(u_re)):
		try:
			u_re[j][i]=u_re[j][i]+mean[u_re[i]]
		except:
			continue




l=len(u_re)
error=0
for i in range(0,l):
	r=len(u_re[i])
	for j in range(0,r):
		try:
			error=((u[i][j]-u_re[i][j])**2)
		except:
			continue
#print k,error

