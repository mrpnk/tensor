import numpy as np

# Reads one number from the file, consumes exactly one extra character.
def _readNextNumber(file) -> int:
	num = 0
	while True:
		c = file.read(1)
		if c == b' ':
			break
		if c == '' or c == b'\x05':
			return "eof"
		num = num*10+int(c)
	return num

# Writes the tensor content to an opened file
def _serialize(file, tensor : np.ndarray):
	s = tensor.shape
	file.write(bytes(str(len(s))+' ',"ansi"))
	for i in s:
		file.write(bytes(str(i)+' ',"ansi"))
	file.write(tensor.tobytes())


# Writes the tensor content to a binary file
def serialize(filename, tensor : np.ndarray):
	with open(filename,"wb") as file:
		_serialize(file,tensor)

# Writes all tensors to a binary file.
def serializeAll(filename, tensors : list):
	with open(filename,"wb") as file:
		for t in tensors:
			_serialize(file,t)


# Loads the first tensor in a binary file.
def deserialize(filename):
	file = open(filename,"rb")
	d = _readNextNumber(file)
	shape=[]
	for i in range(d):
		s = _readNextNumber(file)
		shape.append(s)
	ret = np.fromfile(file,np.float32,-1,'').reshape(tuple(shape))
	file.close()
	return ret

# Returns a list of all tensors in a binary file.
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

