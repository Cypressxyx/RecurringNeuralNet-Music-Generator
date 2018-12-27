''' Recurring Neural Network 
	- Music generator
	- LSTM cells are used to mainta temporal
	dependencies betweeing music notes
'''
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from util.util import print_progress
from util.create_dataset import create_dataset, get_batch
from util.midi_manipulation import noteStateMatrixToMidi


'''
	Generate Song length
	* song length >= minSongLen
'''
def generateData(minSongLen):
	encoded_songs = create_dataset(minSongLen)
	return encoded_songs


'''
	Params:
		- InputVec: input vector placeholder
		- weights:	generated weights
		- biases: 	generated biases
		- return:		rnn graph
'''
def Rnn(inputVec, weights, biases):
	#unstack timeteps into (batch , n inputs)
	#timesteps as length arg, 1 as axis arg
	inputVec = tf.unstack(inputVec, timesteps, 1)
	
	#define a basic lstm cell.
	lstmCell = tf.nn.rnn_cell.LSTMCell(hiddenSize)
	
	#get outputs and states from lstm cell
	outputs, states = rnn.static_rnn(lstmCell, inputVec, dtype=tf.float32)

	#compute hidden layers transformations of the final output of the lstm
	#activation of the final layer: lstmOutput * weights + biases
	lstmOutputs = outputs[-1]
	reccurentNetLogit = tf.matmul(lstmOutputs, weights) + biases
	
	#predict the next note. used to generate a song later
	#generate prob. distrbution over possible notes
	#by computing the softmax of the transformation final output of the lstm
	prediction = tf.nn.softmax(reccurentNetLogit)

	return reccurentNetLogit, prediction
	

''' Init parameters '''
minSongLen = 128
dataSet = generateData(minSongLen)
inputSize = dataSet[0].shape[1] #num of possibile MIDI notes
outputSize = inputSize
hiddenSize = 14 								#number of neurons
eta	= 0.001											#learning rate
epoch = 350											#num of batches during training
batchSize = 64									#num songs per batch
timesteps = 64									#len of song snipet. 
assert timesteps < minSongLen	
	
inputPhSize = [None, timesteps, inputSize] 				#inpput placeholder size
outputPhSize = [None, outputSize]									#output placeholder size
inputVec = tf.placeholder(tf.float32, inputPhSize)
outputVec = tf.placeholder(tf.float32, outputPhSize)

weights = tf.Variable(tf.random_normal([hiddenSize,outputSize]))
biases = tf.Variable(tf.random_normal([outputSize]))

''' Generate an RNN '''
logits, prediction = Rnn(inputVec, weights, biases)

#Loss operation using mean softmax cross entropy loss
lossOperation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels=outputVec))

#training operation
optimizer = tf.train.AdamOptimizer(eta)
trainOperation = optimizer.minimize(lossOperation)

#accuracy
#determine the predcted next note and true next note across training batch

trueNote = tf.argmax(outputVec, 1)
predNote = tf.argmax(prediction, 1)
correctPrediction = tf.equal(trueNote, predNote)

#obtain a value for the accuracy
accuracyOperation = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

#init all variables
init = tf.global_variables_initializer()

session = tf.InteractiveSession()
session.run(init)

#train 
displayStep = 100
for step in range(epoch):
	#get batch
	batchX, batchY = get_batch(dataSet, batchSize, timesteps, inputSize, outputSize)
	
	feed_dict = { inputVec: batchX, outputVec: batchY  }
	session.run(trainOperation, feed_dict = feed_dict)
	
	if step % displayStep == 0:
		loss, acc = session.run([lossOperation, accuracyOperation], feed_dict = feed_dict)
		suffix = "\nStep " + str(step) + ", Minibatch Loss = " + \
							"{:4f}".format(loss) + ", Training Accuray = " + \
							"{:3f}".format(acc) 
		print_progress(step, epoch, barLength = epoch, suffix = suffix)
print("")

GEN_SEED_RAND = True #use random snippet as seed for generating new song

if GEN_SEED_RAND:
	numSongs = len(dataSet)
	ind = np.random.randint(numSongs)
else:
	ind = 14 #blank space chorus

genSong = dataSet[ind][:timesteps].tolist()

#generate song
for i in range(500):
	seed = np.array([genSong[-timesteps:]])
	
	#get prob for next note using seed and trained rnn prediction model
	predictProp = session.run(prediction, feed_dict = {inputVec: seed})
	
	playedNotes = np.zeros(outputSize)
	sampledNote = np.random.choice( range(outputSize), p = predictProp[0])
	playedNotes[sampledNote] = 1
	genSong.append(playedNotes)

noteStateMatrixToMidi(genSong, name="generatedSong0")
noteStateMatrixToMidi(dataSet[ind], name="baseSong0")
noteStateMatrixToMidi(dataSet[ind], name="baseSong0")
print("Saved generated Song! Seed index: {}".format(ind))
