import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import cv2





def normalize(omax, omin, nmax, nmin, ip):
	return (nmax - nmin)/(omax-omin)*(ip-omax)+nmax

dirsave = "Results_SR_4X/"
dirdata = "Data/"
learning_rate=0.0001
epochs = 10000
batchsize = 20
display_step = 5

dimension = 128
n_input = dimension
patch_dimension = 128
n_output = n_input
dim=n_input


ll = 0
hl = 0
incr = batchsize
noisy_image = np.load(dirdata+'pq.npy')
true_image = np.load(dirdata+'hq.npy')
images = np.load(dirdata+'testSR_2X.npy')
widthofimages = np.load(dirdata+'width_testimages_2X.npy')
heightofimages = np.load(dirdata+'height_testimages_2X.npy')
countperimage = np.load(dirdata+'countperimage_2X.npy')
names = np.load(dirdata+'names_2X.npy')

#noisy_image = np.array(noisy_image)
noisy_image = noisy_image.astype(float)
true_image = true_image.astype(float)
images = images.astype(float)

noisy_image = normalize(255.0,0.0, 1.0, 0.0,noisy_image)
true_image=normalize(255.0, 0.0, 1.0, 0.0, true_image)
images = normalize(255.0, 0.0, 1.0, 0.0, images)
'''
noisy_image = noisy_image/255
true_image=true_image/255
images = images/255
'''

print true_image


totsize = true_image.shape[0]
totalsize=images.shape[0]
lowerlimit = 0
higherlimit = 0
def takeAllPatches(image, width, height):
	global patch_dimension
	global lim
	cnt = 0
	i = 0
	recreatedimage = np.zeros((height,width))
	image_array = image
	while (i<height):
		j=0
		while (j<width):
			if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
				rs=i
				re = i+patch_dimension
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension >= height and j+patch_dimension <=width-1:
				rs = height-(patch_dimension)
				re = height
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension <= height-1 and j+patch_dimension >=width:
				rs = i
				re = i+patch_dimension
				cs = width - (patch_dimension)
				ce = width
				
			if i+patch_dimension >= height and j+patch_dimension >=width:
				rs = height-(patch_dimension)
				re = height
				cs = width - (patch_dimension)
				ce = width
				#print 'if-4'
		
			image_toshow = image_array
			recreatedimage[rs:re, cs:ce] = image_toshow[cnt]
			#print cropimage.shape
			cnt = cnt+1
			
			
			j=j+patch_dimension
		i=i+patch_dimension
	return recreatedimage
def nextbatch(batch_i):
	global ll
	global hl
	
	ll = batch_i*batchsize
	hl = batch_i*batchsize + (batchsize)
	#print hl
	#print ll
	tempnoisy = noisy_image[ll:hl].copy()
	tempnormal = true_image[ll:hl].copy()
	tempnormal = tempnormal.reshape((batchsize, dimension,dimension, 1))
	tempnoisy = tempnoisy.reshape((batchsize, dimension/4,dimension/4, 1))
	#print tempnoisy.shap
	return tempnormal, tempnoisy

	
#WEIGHTS AND BIASES

n1 = 32
n2 = 16
n3 = 16

ksize = 3
ksize1 = 5

weightsin = {
	'inthreebythree' : tf.Variable(tf.random_normal([ksize, ksize, 1, n1], stddev = 0.1))

}
biasesin = {
	'inthreebythree' : tf.Variable(tf.random_normal([n1], stddev = 0.1))

}

weightsdimred = {
	'inonebyone' : tf.Variable(tf.random_normal([1, 1, n1, n3], stddev = 0.1))

}
biasesdimred = {
	'inonebyone' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}

weightsrec = {
	'inonebyone' : tf.Variable(tf.random_normal([1, 1, n3, 1], stddev = 0.1))

}
biasesrec = {
	'inonebyone' : tf.Variable(tf.random_normal([1], stddev = 0.1))

}

#WEIGHTS 1 
weightsupb1 = {
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesupb1 = {
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}
weightsdwb1 = {
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesdwb1 = {
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}

#WEIGHTS 2 

weightsupb2 = {
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesupb2 = {
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}
weightsdwb2 = {
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesdwb2 = {
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}

#WEIGHTS 3 
weightsupb3 = {
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesupb3 = {
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}



#WEIGHTS 4 
weightsupb4 = {
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesupb4 = {
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}
weightsdwb4 = {
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesdwb4 = {
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}

#WEIGHTS 5

weightsupb5 = {
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesupb5 = {
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}
weightsdwb5 = {
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesdwb5 = {
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}

#WEIGHTS 6 
weightsupb6 = {
	'us1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'ds1' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1)),
	'us2' : tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev = 0.1))

}

biasesupb6 = {
	'bus1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bds1' : tf.Variable(tf.random_normal([n3], stddev = 0.1)),
	'bus2' : tf.Variable(tf.random_normal([n3], stddev = 0.1))

}



def leaky_relu(x, alpha=0.2):
	return tf.maximum(x, alpha*x)

def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1,1,1,1], padding='SAME'),b))



def caeUSB(_X, _W, _b, _keepprob, alpha = 0.2):
	_input_r = _X

	_h0 = tf.add(tf.nn.conv2d_transpose(_input_r, _W['us1'], tf.stack([tf.shape(_input_r)[0],tf.shape(_input_r)[1]*2,tf.shape(_input_r)[2]*2,n3]), strides = [1,2,2,1], padding = 'SAME'), _b['bus1'])
	_ch0 = leaky_relu(_h0)
	_ch0 = tf.nn.dropout(_ch0, _keepprob)

	_l0 = tf.add(tf.nn.conv2d(_ch0, _W['ds1'], strides = [1,2,2,1], padding='SAME'), _b['bds1'])
	_cl0 = leaky_relu(_l0)
	_cl0 = tf.nn.dropout(_cl0, _keepprob)

	_e0 = _cl0 - _input_r

	_h1 = tf.add(tf.nn.conv2d_transpose(_e0, _W['us2'], tf.stack([tf.shape(_input_r)[0],tf.shape(_input_r)[1]*2,tf.shape(_input_r)[2]*2,n3]), strides = [1,2,2,1], padding = 'SAME'), _b['bus2'])
	_ch1 = leaky_relu(_h1)
	_ch1 = tf.nn.dropout(_ch1, _keepprob)

	_ht = _ch0 + _ch1
	_out = _ht

	return _out


def caeDSB(_X, _W, _b, _keepprob, alpha = 0.2):
	_input_r = _X

	_l0 = tf.add(tf.nn.conv2d(_input_r, _W['ds1'], strides = [1,2,2,1], padding='SAME'), _b['bds1'])
	_cl0 = leaky_relu(_l0)
	_cl0 = tf.nn.dropout(_cl0, _keepprob)

	_h0 = tf.add(tf.nn.conv2d_transpose(_cl0, _W['us1'], tf.stack([tf.shape(_input_r)[0],tf.shape(_input_r)[1],tf.shape(_input_r)[2],n3]), strides = [1,2,2,1], padding = 'SAME'), _b['bus1'])
	_ch0 = leaky_relu(_h0)
	_ch0 = tf.nn.dropout(_ch0, _keepprob)

	_e0 = _ch0 - _input_r

	_l1 = tf.add(tf.nn.conv2d(_ch0, _W['ds2'], strides = [1,2,2,1], padding='SAME'), _b['bds2'])
	_cl1 = leaky_relu(_l1)
	_cl1 = tf.nn.dropout(_cl1, _keepprob)

	_lt = _cl0 + _cl1
	_out = _lt

	return _out



def calculateL2loss(im1, im2):
	return tf.reduce_mean(tf.square(im1-im2))

def calculateL1loss(im1, im2):
	return tf.reduce_sum(tf.abs(im1-im2))

def optimize(cost, learning_rate = 0.0001):
	return tf.train.AdamOptimizer(learning_rate).minimize(cost)



x = tf.placeholder(tf.float32, [None, None, None, 1])
y = tf.placeholder(tf.float32, [None, None, None, 1])

keepprob = tf.placeholder(tf.float32)

op_l1 = conv2d(x, weightsin['inthreebythree'], biasesin['inthreebythree'])

op_l2 = conv2d(op_l1, weightsdimred['inonebyone'], biasesdimred['inonebyone'])

op_ub1 = caeUSB(op_l2, weightsupb1, biasesupb1, keepprob)

op_db1 = caeDSB(op_ub1, weightsdwb1, biasesdwb1, keepprob)

op_ub2 = caeUSB(op_db1, weightsupb2, biasesupb2, keepprob)

op_db2 = caeDSB(op_ub2, weightsdwb2, biasesdwb2, keepprob)

op_ub3 = caeUSB(op_db2, weightsupb3, biasesupb3, keepprob)

op_ub4 = caeUSB(op_ub3, weightsupb4, biasesupb4, keepprob)

op_db4 = caeDSB(op_ub4, weightsdwb4, biasesdwb4, keepprob)

op_ub5 = caeUSB(op_db4, weightsupb5, biasesupb5, keepprob)

op_db5 = caeDSB(op_ub5, weightsdwb5, biasesdwb5, keepprob)

op_ub6 = caeUSB(op_db5, weightsupb6, biasesupb6, keepprob)

pred =  conv2d(op_ub6, weightsrec['inonebyone'], biasesrec['inonebyone'])


cost1 = calculateL2loss(pred, y)


optm1 = optimize(cost1, learning_rate)



print ("Network ready")


init = tf.global_variables_initializer()

print("All functions ready")
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)


print("Start training")

for epoch_i in range(epochs):
	num_batch = int(totsize/(batchsize))
	for batch_i in range(num_batch):
		batch_xs, batch_xs_noisy = nextbatch(batch_i)
		'''
		cv2.imshow("to", batch_xs_noisy[3])
		cv2.waitKey(100)

		cv2.imshow("to1", batch_xs[3])
		cv2.waitKey(100)
		'''
		sess.run([optm1], feed_dict = {x:batch_xs_noisy, y:batch_xs, keepprob:1})
		
	ll=0
	hl=0
	cost = sess.run(cost1, feed_dict={x : batch_xs_noisy, y: batch_xs, keepprob :1})
	print("[%02d/%02d] Cost1: %.6f" % (epoch_i, epochs, cost))
	if cost < 0.0005:
		learning_rate = 0.00001
	elif cost < 0.0002:
		learning_rate = 0.000001

	if epoch_i % display_step == 0 or epoch_i == epochs - 1:

		saver.save(sess, "logs/4XSR.ckpt")
		
		for i in range(countperimage.shape[0]):
			higherlimit = int(higherlimit+countperimage[i])
			allpatchesofanimage = images[lowerlimit:higherlimit].copy()
			lowerlimit = int(lowerlimit + countperimage[i])
			print countperimage[i]
			reconstructedimage = np.zeros([int(countperimage[i]), patch_dimension, patch_dimension])
			for j in range(int(countperimage[i])):
				recon = sess.run(pred, feed_dict = {x:allpatchesofanimage[j].reshape(1, patch_dimension/4, patch_dimension/4, 1), keepprob:1.})
				recon = recon.reshape((1, patch_dimension, patch_dimension))
				reconstructedimage[j]=recon

			recreatedimage = takeAllPatches(reconstructedimage, int(widthofimages[i]*4), int(heightofimages[i]*4))
			recreatedimage = normalize(1.0, 0.0, 255.0, 0.0, recreatedimage)
			cv2.imwrite(dirsave+names[i], recreatedimage)
		lowerlimit=0
		higherlimit=0
		
		


		
		









































