import functools
import Queue
import threading
import glob 
import os
def async_prefetch_wrapper(iterable, buffer=10):
	"""
	wraps an iterater such that it produces items in the background
	uses a bounded queue to limit memory consumption
	"""
	done = object()
	def worker(q,it):
		for item in it:
			q.put(item)
		q.put(done)
	# launch a thread to fetch the items in the background
	queue = Queue.Queue(buffer)
	it = iter(iterable)
	thread = threading.Thread(target=worker, args=(queue, it))
	thread.daemon = True
	thread.start()
	# pull the items of the queue as requested
	while True:
		item = queue.get()
		if item == done:
			return
		else:
			yield item
 
def async_prefetch(func):
	"""
	decorator to make generator functions fetch items in the background
	"""
	@functools.wraps(func)
	def wrapper(*args, **kwds):
		return async_prefetch_wrapper( func(*args, **kwds) )
	return wrapper

 
 
def test_setup():
	files = []
	lines = 1000000
	for i in xrange(100):
		filename = "tempfile%d.txt"%i
		files.append(filename)
		with open(filename, "w") as f:
			f.write( ("%d\n"%i)*lines )
	return files
 
def test_cleanup():
	for f in glob.glob("tempfile*.txt"):
		os.unlink(f)
 

#@async_prefetch
def contents(iterable):
	for filename in iterable:
		with open(filename, "rb") as f:
			contents = f.read()
		yield contents
 
def test():
	files = test_setup()
	for c in contents(files):
		hashlib.md5(c).digest()
	test_cleanup()
 
from timeit import Timer
t = Timer("test()", "from __main__ import test; gc.enable()")
print t.repeat(5, 1)