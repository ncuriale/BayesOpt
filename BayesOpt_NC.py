import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random
import time
pl.ion()

#import latinHypercube

class RBFKernel(object):
	"""
	RBF Kernel Function
	"""
	def __init__(self,*args):
		self.thetas = args

	def __call__(self,x,y):
		exponential = self.thetas[0] * np.exp( -0.5 * self.thetas[1] * np.sum( (x - y)**2 ) )
		return exponential

class OrnsteinKernel(object):
	"""
	Ornstein-Uhlenbeck process kernel.
	"""
	def __init__(self,theta):
		self.theta = theta

	def __call__(self,x,y):
		return np.exp(-self.theta * np.sum(np.abs(x-y)))

def covariance(kernel, data):

	K = np.empty(( len(data[0]), len(data[0]) ))

	for j in range(len(data[0])):
		for i in range(len(data[0])):
			K[j,i] = kernel(data[:,j], data[:,i])

	return K

def train(data, kernel):
	C = covariance(kernel,data)
	return C

def fit(x, dataX, dataY, C, kernel):

	#Inverse of covariance
	Cinv = np.linalg.inv(C)

	#Calculate new kernels
	kxx = kernel(x,x)
	kxy = np.empty(( len(dataX[0]) ))
	for j in range(len(dataX[0])):
		kxy[j]=kernel(x, dataX[:,j])

	mu = np.dot(np.dot(Cinv, kxy), dataY.T)
	sigma = kxx - np.dot(np.dot(kxy, Cinv),kxy)

	return mu, sigma

def utility(x, y, sig, mode, kappa=2):

	#upper confidence bound
	if (mode=='max'):
		val = y + np.asarray(kappa)*sig

	#lower confidence bound
	if (mode=='min'):
		val = y - np.asarray(kappa)*sig

	return val

def nextUtility(x, y, util, ndim, mode):

	### This returns the array of x-values where y is maximized through various methods

	if(mode=='max'):

		ynext = np.array([[ np.max(util) ]])
		xnext = np.expand_dims( x[:,np.argmax(util)], axis=1)
		'''
		ynext = np.array([[ np.max(util) ]])
		indx = np.unravel_index(np.argmax(util), x[0].shape)

		xnext = np.empty(( ndim ))
		for i in range(ndim):
			temp=x[i]
			for _ in range(ndim):
				temp=temp[ indx[i] ]

			xnext[i] = np.array([[ temp ]])

		xnext = np.expand_dims( xnext, axis=1 )
		'''

	if(mode=='min'):

		ynext = np.array([[ np.min(util) ]])
		xnext = np.expand_dims( x[:,np.argmin(util)], axis=1)
		'''
		ynext = np.array([[ np.min(util) ]])
		indx = np.unravel_index(np.argmin(util), x[0].shape)

		xnext = np.empty(( ndim ))
		for i in range(ndim):
			temp=x[i]
			for _ in range(ndim):
				temp=temp[ indx[i] ]

			xnext[i] = np.array([[ temp ]])

		xnext = np.expand_dims( xnext, axis=1 )
		'''

	return xnext, ynext

def nextRandUtility(util, ndim, ndx, lo, hi, mode):

	# Set up rand size
	ynext=None
	size=(250,ndim)
	x_rands = np.zeros(( size ))
	xstar = np.zeros(( ndim ))

	# Utility size set up
	utilSize, _ = sizeMesh(ndx, ndim)
	utilVals = np.reshape(util, utilSize )

	# Explore the space randomly
	for i in range(ndim):
		x_rands[:,[i]] = np.random.uniform(lo[i], hi[i], size=(250,1))

	# Go through all random vectors
	for x_i in x_rands:
		'''
		# Find the minimum of minus the acquisition function
		res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
				x_try.reshape(1, -1),
                     bounds=bounds,
                     method="L-BFGS-B")
		'''

		# Search all dimensions of vector to find corresponding index
		for i in range(ndim):
			dx=( float(hi[i]) - float(lo[i]) ) / ( float(ndx) - 1 )
			for j in range(ndx):
				if ( x_i[i] < (lo[i] + j*dx) ):
					xstar[i]=j
					break

		# Set utility value corresponding to index
		y_i=utilVals[tuple(xstar.astype(int))]

		# Store it if better than previous minimum(maximum).
		if (mode=='max'):
			if ynext is None or y_i >= ynext:
				xnext = x_i
				ynext = y_i
		if (mode=='min'):
			if ynext is None or y_i <= ynext:
				xnext = x_i
				ynext = y_i

	xnext = np.expand_dims( xnext, axis=1 )

	return xnext, ynext

def findBest(X, Y, y_best, mode):

	# 1D array shifting function
	def shift(array, key):
		return np.concatenate((array[-key:], array[:-key]), axis=0)

	if(mode=='min'):
		# FIND, PRINT AND SAVE MAX FROM DATASET
		i_best = np.argmin(Y)
		x_best = X[:,i_best]
		y_best[-1] = np.min(Y)
		y_best=shift(y_best,1)

		print ('Minimum: --> index:', i_best, 'X:', x_best, 'Y:', y_best[0])
		print ('')

	if(mode=='max'):
		# FIND, PRINT AND SAVE MAX FROM DATASET
		i_best = np.argmax(Y)
		x_best = X[:,i_best]
		y_best[-1] = np.max(Y)
		y_best=shift(y_best,1)

		print ('Maximum: --> index:', i_best, 'X:', x_best, 'Y:', y_best[0])
		print ('')

	return i_best, x_best, y_best

def checkBest(X, Y, i_best, x_best, y_best, y_diff, mode):

	if(mode=='min'):
		# Update min value
		if y_best[0] < np.min(Y):
			i_best, x_best, y_best = findBest(X,Y,y_best,mode)
			# Calculate differences in array
			for id in range(0,len(y_diff)-1):
				y_diff[id]=abs( float(y_best[id]) - float(y_best[id+1]) )

	if(mode=='max'):
		# Update max value
		if y_best[0] < np.max(Y):
			i_best, x_best, y_best = findBest(X,Y,y_best,mode)
			# Calculate differences in array
			for id in range(0,len(y_diff)-1):
				y_diff[id]=abs( float(y_best[id]) - float(y_best[id+1]) )

	return i_best, x_best, y_best, y_diff

def fillMesh(dx, ndx, ndim, lo ):

	####
	#### Creates a mesh that has 'ndim' dimensions and 'ndx' points in each dimension
	####

	size=()
	for _ in range(ndim):
		size += (ndx,)

	vec=[]
	arr = np.zeros(( ndx ))

	# Fill out discretized space
	for j in range(ndx):
		arr[j] = lo + j*dx

	if(ndim>1):
		# Expand discretized space in 1D for all dimensions
		for _ in range(ndx**(ndim-1)):
			vec=np.append(vec,arr)
		
		# Reshape discretized space into n-dim array
		vec=np.reshape( vec, size )

	# Ensure this condition works
	else:
		vec=np.append(vec,arr)

	return vec

def fillMesh_recursive(dx, ndx, ndim, lo, dimLevel=-1, j=0 ):

	#### Creates a mesh that has 'ndim' dimensions and 'ndx' points in each dimension
	####
	#### Requires dimLevel=-1 for initial call
	#### Do not require j to be any value on intial call
	####

	if dimLevel==(ndim-1):
		vec = lo + j*dx

	else:
		dimLevel+=1

		size=()
		for _ in range(ndim-dimLevel):
			size += (ndx,)
		vec = np.zeros(( size ))

		for j in range(ndx):
			vec[j] = fillMesh_recursive( dx, ndx, ndim, lo, dimLevel, j )

	return vec

def sizeMesh(ndx, ndim):

	## DYNAMIC ARRAY SIZE BASED ON: ndim, ndx
	size=()
	for _ in range(ndim):
		size += (ndx,)

	xsize = (ndim,) + size

	return size, xsize

def defineMesh(ndim, ndx, Xlo, Xhi):

	## DYNAMIC SIZE BASED ON: ndim, ndx
	size, xsize = sizeMesh(ndx, ndim)
	x_prob=np.zeros(( xsize ))
	y_prob = np.array([ np.zeros(( size )) ])
	sigma = np.array([ np.zeros(( size )) ])

	# x_prob needs to be expanded to a mesh in all dimensions
	# Loop to split dimensions into subarrays
	# Each sub-array is different dim --- x,y,z,etc
	for i in range(ndim):

		### Discretization of grid in specific dimension
		dx=( float(Xhi[i]) - float(Xlo[i]) ) / ( float(ndx) - 1 )

		### Call recursive function to make n-dimensional grid
		x_prob[i] = fillMesh( dx, ndx, ndim, Xlo[i] )

	return x_prob, y_prob, sigma

def reorderMesh(mesh, mesh_shape, ndim, ndx):

	# Array indexes for moveaxis shifting
	#indxSeq=np.zeros(( ndim )).astype(int)
	#indxVals=np.zeros(( ndim )).astype(int)
	#for i in range(ndim):
	#    indxVals[i]=i
	#    indxSeq[i]=-i-1

	# 1D array shifting function
	#def shift(array, key):
	#    return np.concatenate((array[-key:], array[:-key]), axis=0)

	# 2D array shifting function
	def shift_2D(array, key):
		return np.concatenate((array[:, -key:], array[:, :-key]), axis=1)

	# Sets indices for 'ndim' dimensions array
	size, _ = sizeMesh(ndx, ndim)
	indxs=np.zeros(( ndx**ndim,ndim )).astype(int)
	j=0
	for i in np.ndindex(size):
		indxs[j]=i
		j+=1

	# Go through all dimensions, shift indexes, then reorder array
	for k in range(ndim):
		#temp=np.moveaxis(mesh[k], indxVals, indxSeq)
		#indxSeq=shift(indxSeq,1)
		indxTemp=shift_2D(indxs, k)
		j=0
		for i in indxTemp:
			mesh_shape[k,j]=mesh[k][tuple(i)]
			j+=1

	return mesh_shape

def plotInfo(x_prob,y_prob,sigma,X,Y,x_util,y_util,utilVals,ndx,ndim,mode,j=0):

	# Reshape array to original
	size, xsize = sizeMesh(ndx, ndim)
	x_prob = np.reshape(x_prob, xsize )
	y_prob = np.reshape(y_prob, size )
	sigma = np.reshape(sigma, size )
	utilVals = np.reshape(utilVals, size )

	if(ndim==1):
		## PLOT 1D ##
		plt.figure(j)

		ax1=plt.subplot(211)
		plt.errorbar(x_prob[0,:], y_prob, yerr=sigma, capsize=0)
		plt.plot(X[0,:], Y[0,:], 'ko')	#plot all points
		axes = plt.gca()
		axes.set_xlim([min(x_prob[0,:])-1,max(x_prob[0,:])+1])
		ax1.set_title('Mean Value')

		ax2=plt.subplot(212)
		plt.plot(x_prob[0,:], utilVals, 'k-')
		plt.plot(x_util[0,:], y_util, 'ro')	#plot recent point
		axes = plt.gca()
		axes.set_xlim([min(x_prob[0,:])-1,max(x_prob[0,:])+1])
		ax2.set_title('Utility Function')

	if(ndim==2):
		## PLOT 2D ##
		plt.figure(j)

		# Mean Value
		ax1=plt.subplot(221)
		cb1=plt.contourf(x_prob[0], x_prob[1], y_prob)	#plot mean contour
		plt.plot(X[0], X[1], 'ko')	#plot all points
		plt.plot(X[0,-1], X[1,-1], 'ro')	#plot most recent point
		ax1.set_title('Mean Value')
		plt.colorbar(cb1)
		axes = plt.gca()

		# Target Func
		ax2=plt.subplot(222)
		funcVals=np.empty_like(y_prob)
		for i in range(ndx):
			for j in range(ndx):
				x_i = [ x_prob[0][i,j], x_prob[1][i,j] ]
				funcVals[i,j] = func(x_i,mode)
		cb2=plt.contourf(x_prob[0], x_prob[1], funcVals)	#plot function contour
		plt.plot(X[0], X[1], 'ko')	#plot all points
		plt.plot(X[0,-1], X[1,-1], 'ro')	#plot most recent point
		ax2.set_title('Target Function')
		plt.colorbar(cb2)

		# Variance Value
		ax3=plt.subplot(223)
		cb3=plt.contourf(x_prob[0], x_prob[1], sigma)	#plot sigma contour
		plt.plot(X[0], X[1], 'ko')	#plot all points
		plt.plot(X[0,-1], X[1,-1], 'ro')	#plot most recent point
		ax3.set_title('Variance Value')
		plt.colorbar(cb3)

		# Utility Func
		ax4=plt.subplot(224)
		cb4=plt.contourf(x_prob[0], x_prob[1], utilVals)	#plot utility contour
		plt.plot(X[0], X[1], 'ko')	#plot all points
		plt.plot(X[0,-1], X[1,-1], 'ro')	#plot most recent point
		plt.axvline(x=X[0,-1], color='k', linestyle='--')	#plot most recent point
		plt.axhline(y=X[1,-1], color='k', linestyle='--')	#plot most recent point
		ax4.set_title('Utility Function')
		plt.colorbar(cb4)

def sampling( lo, hi, npts, ndim, samp):

	if (samp=='lhc'):
		samples = np.empty(( ndim, npts ))	
		segSize = 1/float(npts)
		for i in range(ndim):
			isamples = np.zeros(( npts ))
			for j in range(npts):
				segMin = float(j) * segSize
				point = segMin + (np.random.rand() * segSize)
				isamples[j] = (point * (hi[i] - lo[i])) + lo[i]

			samples[i,:]=np.random.permutation(isamples)

	return samples

def func(x, mode):

	if (mode=='max'):
		sgn=-1
	if (mode=='min'):
		sgn=1

	#f = x[0]**2 + x[1]**2
	#f = ( 4 - 2.1*x[0]**2 +  (1/3)*x[0]**4 )*x[0]**2 +  x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2 #camelback
	#f = ( x[1] - (5.1/(4*3.14**2))*x[0]**2 - (5/3.14)*x[0] - 6 )**2 + 10*( 1 - (1/(8*3.14)))*np.cos(x[0]) + 10 #branin
	#f = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2  #rosenbrock
	f=x[0]**2 #+ x[1]**2 + x[2]**2 + x[3]**2

	return sgn*np.array([f])

def main():

	###### Kernel Options ######
	#kernel = OrnsteinKernel(1.0)
	kernel = RBFKernel(1.0, 1)

	###### PARAMETERS ######
	ndx=10
	ndim=2
	niter=20
	tol=1e-5
	mode='min'
	samp='lhc'
	npts=1
	plotOn=True

	###### BOUNDS ######
	t=1
	Xlo = [-t, -t, -t, -t, -t, -t, -t, -t, -t, -t, -t, -t, -t, -t, -t, -t]
	Xhi = [ t,  t,  t,  t,  t,  t,  t,  t,  t,  t,  t,  t,  t,  t,  t,  t]

	###### PRIOR DATA ######
	#X = np.array([
	#	[-t, -t, -t,  0,  0,  0,  t,  t,  t],
	#	[-t,  0,  t, -t,  0,  t, -t,  0,  t],
	#	[-8, -8, -8,  0,  0,  0,  8,  8,  8],
		#[-8,  0,  8, -8,  0,  8, -8,  0,  8],
	#	[0, -8, 6.5],
	#	[0, -2, 1.5]
	#	])
	#Y = np.array([
	#    [1, 0.5, -0.5]
	#    ])
	X = sampling( Xlo, Xhi, npts, ndim, samp )
	X = X[0:ndim,0:npts]

	Y = np.empty(( len(X[0]) ))
	for i in range(len(X[0])):
		Y[i] = func(X[:,i],mode)
	Y = Y[0:npts]
	Y = np.expand_dims( Y, axis=0 )


	###### NEW POINTS ######
	x_func = np.array([[-1,-1,0,0]]).T
	x_func = x_func[0:ndim]

	####### DISCRETIZE MESH & DEFINE ARRAYS ######
	x_prob, y_prob, sigma = defineMesh( ndim, ndx, Xlo, Xhi )
	# Reshape array to predict with 'ndim' coordinates
	y_prob_reshape=np.reshape(y_prob, (1, ndx**ndim) )
	sigma_reshape=np.reshape(sigma, (1, ndx**ndim) )
	#### Reshape/reorder x_prob at the same time
	x_prob_shape=np.empty(( ndim, ndx**ndim ))
	x_prob_reshape=reorderMesh(x_prob, x_prob_shape, ndim, ndx)

	######INITIAL BEST OF DATASET ######
	y_best=np.random.rand(4,1).astype(float)
	y_diff=np.zeros((4,1)).astype(float)
	i_best, x_best, y_best = findBest(X,Y,y_best,mode)
	# Calculate differences in array
	for id in range(0,len(y_diff)-1):
		y_diff[id]=abs(y_best[id]-y_best[id+1])


	######################################
	#------------------------------------#
	#----- ITERATE FOR OPTIMIZATION -----#
	#------------------------------------#
	######################################

	for j in range(0,niter):

		print (j,'-- Next Point:',)
		for cnt in x_func:
			print (cnt,)

		# Check if already tested
		if np.any((X - x_func).sum(axis=0) == 0):
			print ('    Already tested -- switch point randomly')
			x_func =np.array([np.random.uniform(Xlo[0:ndim],Xhi[0:ndim])]).T
			print ('    New point:',)
			for cnt in x_func:
				print (cnt,)
			print ('')

		# Evaluate function with new vector
		y_func = func(x_func,mode)
		print ('---- Function Value:',y_func)
		print ('')
		print ('')

		# Add points to x and y then re-evaluate
		X = np.hstack(( X, x_func ))
		Y = np.hstack(( Y, y_func ))

		# Update best value
		i_best, x_best, y_best, y_diff = checkBest(X, Y, i_best, x_best, y_best, y_diff, mode)

		# Train existing points to get cov
		cov = train(X, kernel)

		##################################################
		### Explore GP regression in the target domain ###
		##################################################

		# Loop through all coordinates to predict values
		for i in range(ndx**ndim):
			predictions = fit(x_prob_reshape[:,i],X,Y,cov,kernel)
			y_prob_reshape[:,i] = predictions[0]
			sigma_reshape[:,i] = predictions[1]

		# Evaluate the utility function and find maximum position
		utilVals = utility(x_prob_reshape, y_prob_reshape, sigma_reshape, mode)
		#x_util, y_util = nextRandUtility(utilVals, ndim, ndx, Xlo, Xhi, mode )
		x_util, y_util = nextUtility( x_prob_reshape, y_prob_reshape, utilVals, ndim, mode )
		print('Utility:', y_util)

		# Set point for next loop
		x_func = x_util

		if(plotOn):
			plotInfo(x_prob_reshape,y_prob_reshape,sigma_reshape,X,Y,x_util,y_util,utilVals,ndx,ndim,mode,j)
			pl.draw()
			time.sleep(10)
			pl.close()

		# Check convergence
		if ( np.all(y_diff < tol) ):
			break

	# Print Optimal Values
	print ('Optimal Values:')
	print ('---- index:',i_best)
	print ('---- X:',x_best)
	print ('---- Y:',y_best)

	plotInfo(x_prob_reshape,y_prob_reshape,sigma_reshape,X,Y,x_util,y_util,utilVals,ndx,ndim,mode,-1)
	plt.show()

if __name__ == '__main__':
     main()





