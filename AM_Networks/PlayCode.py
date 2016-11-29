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