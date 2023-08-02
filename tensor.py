import numpy as np

def readNextNumber(file) -> int:
	num = 0
	while True:
		c = file.read(1)
		if c == b' ':
			break
		if c == '' or c == b'\x05':
			return "eof"
		num = num*10+int(c)
	return num

def serialize(file, tensor : np.ndarray):
	s = tensor.shape
	file.write(bytes(str(len(s))+' ',"ansi"))
	for i in s:
		file.write(bytes(str(i)+' ',"ansi"))
	file.write(tensor.tobytes())

# Write to file.
def serializeAll(filename, tensorList):
	with open(filename,"wb") as file:
		for t in tensorList:
			serialize(file,t)

# Load from file.
def deserialize(filename):
	file = open(filename,"rb")
	d = readNextNumber(file)
	shape=[]
	for i in range(d):
		s = readNextNumber(file)
		shape.append(s)
	ret = np.fromfile(file,np.float32,-1,'').reshape(tuple(shape))
	file.close()
	return ret


def deserializeAll(filename):
	ret=[]
	file = open(filename,"rb")
	while True:
		d = readNextNumber(file)
		if d == "eof":
			break
		shape=[]
		vol = 1
		for i in range(d):
			s = readNextNumber(file)
			shape.append(s)
			vol *= s
		ret.append(np.fromfile(file,np.float32,vol,'').reshape(tuple(shape)))
	file.close()
	return ret

#%%
