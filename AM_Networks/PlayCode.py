import glob
import numpy as np
import cv2


DataDir='/Users/talathi1/Work/DataSets/AM_Project/Resized_256x256'
Img_List=glob.glob('%s/*.jpg'%DataDir)

np.random.seed(100)

#Select Validation Set Randomly
rnd_img_list=np.random.randint(0,len(Img_List),2000)

X_train=[];y_train=[];
X_test=[];y_test=[]
for i in range(len(Img_List)):
	img=cv2.imread(Img_List[i])
	label_str=Img_List[i].split('/')[-1].split('_')[0]
	label=0 if 'Bad' in label_str else (1 if 'Good' in label_str else 2)
	img_resize=cv2.resize(img,(224,224),interpolation = cv2.INTER_LINEAR)
	img_resize_rescale=1.0*img_resize/img_resize.max()	
	if i in rnd_img_list:
		X_test.append(img_resize_rescale)
		y_test.append(label)
	else:
		X_train.append(img_resize_rescale)
		y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)



### Defining AM data generator
def AM_Data_Generator(Img_List,batch_size=32,idx=0):
	Img_Array=np.array(Img_List)
	X=np.zeros((batch_size,224,224,3));y=np.zeros(batch_size)
	while True:
		ind=np.arange(idx,batch_size+idx)
		subset_Img_Array=Img_Array[ind]
		for i in range(len(subset_Img_Array)):
			img=cv2.imread(subset_Img_Array[i])
			label_str=subset_Img_Array[i].split('/')[-1].split('_')[0]
			label=0 if 'Bad' in label_str else (1 if 'Good' in label_str else 2)
			print label_str,label,subset_Img_Array[i]
			img_resize=cv2.resize(img,(224,224),interpolation = cv2.INTER_LINEAR)
			img_resize_rescale=1.0*img_resize/img_resize.max()	
			X[i,:,:,:]=img_resize_rescale
			y[i]=label
		yield X,y
		idx=idx+batch_size

## Defining a good data generator to select subset of array elements
def Test_Generator(X,N,idx):
	while idx<=len(X):
		ind=np.arange(idx,N+idx)
		yield X[ind],idx
		idx=idx+N


X=np.random.randint(0,100,50)
y=np.random.randn(50)



def take(count,iterable):
	counter=0
	for item in iterable:
		if counter==count:
			return
		counter+=1
		yield item

def run_take(count,items):
	for item in take(count,items):
		print(item)

def take(count,iterable):
	counter=0
	for item in iterable:
		if counter==count:
			return
		counter+=1
		yield item

def run_take(count,items):
	for item in take(count,items):
		print(item)


import threading
class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def next(self):
        self.lock.acquire()
        try:
            return self.it.next()
        finally:
            self.lock.release()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return LockedIterator(f(*a, **kw))
    return g


## Test generatore read spead
import itertools as it
datagen=AH.AM_Data_Generator(imglist,batch_size=1000)
tic=time.clock()
for _ in it.count():
	try:
		X,y=next(datagen)
		print len(X)
	except StopIteration:
		break
toc=time.clock()
print toc-tic

testgen=AH.BackgroundGenerator(AH.AM_Data_Generator(imglist,batch_size=1000))
tic=time.clock()
for _ in it.count():
	try:
		X,y=testgen.next()
		print len(X)
	except StopIteration:
		break
toc=time.clock()
print toc-tic	


### Alternative generator
class MyGen():
    def __init__(self,generator):
        self.queue = Queue.Queue()
        # Put a first element into the queue, and initialize our thread
        self.generator = generator
        self.t = threading.Thread(target=self.worker, args=(self.queue, self.generator))
        self.t.start()

    def __iter__(self):
        return self

    def worker(self, queue, generator):
        queue.put(generator)

    def __del__(self):
        self.stop()

    def stop(self):
        while True: # Flush the queue
            try:
                self.queue.get(False)
            except Queue.Empty:
                break
        self.t.join()

    def next(self):
        # Start a thread to compute the next next.
        self.t.join()
        self.i += 1
        self.t = threading.Thread(target=self.worker, args=(self.queue, self.i))
        self.t.start()

        # Now deliver the already-queued element
        while True:
            try:
                print "request at", time.time()
                obj = self.queue.get(False)
                self.queue.task_done()
                return obj
            except Queue.Empty:
                pass
            time.sleep(.001)