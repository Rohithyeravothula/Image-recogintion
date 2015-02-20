import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
import glob
import math
from sklearn import svm
import pybrain
from pybrain.tools.shortcuts import*
from pybrain.supervised.trainers import BackpropTrainer	
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
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
			u[j][i]=u[j][i]-mean[i]
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
# we have eigvector.shape

eigvector=eigvector[0:180]  # consider 180 eigenectors
eigvector=eigvector.T
l=len(eigvector)
for i in range(0,l):
	eigvector[i]=eigvector[i]/np.linalg.norm(eigvector[i])
#print eigvector.shape,w.shape
u_re=np.dot(wt,eigvector)
#print eigvector.shape,u_re.shape
u_re=np.dot(u_re,eigvector.T)
#print u_re.shape
u_re=u_re.T

# display images

# done displayinh images


# implementing SVM

cls=svm.SVC()
X=np.dot(wt,eigvector)
X=X.T
#print "shape",wt.shape,eigvector.shape,X.shape
Y=[]
for i in range(0,27):
	Y.append(1)
for i in range(0,52):
	Y.append(2)
for i in range(0,41):
	Y.append(3)
for i in range(0,35):
	Y.append(4)
for i in range(0,25):
	Y.append(0)
#print X.shape,Y.shape
#Y=np.asarray(Y)
#print X.shape,Y.shape
#print X
cls.fit(X,Y)
#print cls
p=glob.glob("testdata/*.jpg")
l=len(p)
#print l
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
l=len(u)
#print l
r=len(u[0])
#print "r",r,l
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
#print len(mean)
for i in range(0,len(u[0])):
	for j in range(0,len(u)):
		try:
			u[j][i]=u[j][i]-mean[i]
		except:
			continue
#print len(u)
# u contains centralised data
#print len(u)
#print w_t.shape
w_t=np.array(u[0])
l=len(u)
for i in range(1,l):
	try:
		w_t=np.vstack([w_t,np.array(u[i])])
	except:
		continue
w_tt=w_t.T 
x_t=np.dot(w_t,w_tt)
x_t=x_t/x_t.shape[0]
     # x is covarianve matrix
result=np.linalg.eig(x_t)
eigvector_t=result[1]
output=cls.predict(w_t)
error_svm=0
l=len(output)
for i in range(0,l):
	if output[i]!=Y[i]:
		error_svm=error_svm+1
print "error percentage",str((error_svm/80.0)*100)


#print eigvector_t.shape
#w_t=w_t**2
#print cls 


# input to SVM






# neural network

net=buildNetwork(211, 100, 5, bias=True)
#trainer = BackpropTrainer(net,w)
