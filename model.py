from sklearn.datasets import load_svmlight_file
import numpy as np
import numpy.linalg as LA
import time
from scipy.sparse import *
import collections
from multiprocessing import Pool
import sys
import os
import  cPickle as pickle

def main(argv):
	data_size = argv[0]
	idir = "../data/" 
	odir = idir + data_size + "/"

	if not os.path.exists(odir):
		os.makedirs(odir)

	X_train,Y_train = load_svmlight_file(idir + "train-" + data_size + ".csv", n_features=1617899, multilabel=True)
	print "######### Reading of train data done. Size is: ", np.shape(X_train), " #############"	
	idf_train = get_variable(odir,"idf", lambda:get_idf(X_train))
	tfidf_train = get_variable(odir,"tfidf_train", lambda:get_tfidf(X_train, idf_train))

	X_test,Y_test = load_svmlight_file(idir + "test-" + data_size + ".csv", n_features=1617899, multilabel=True)
	print "######### Reading of test data done. Size is: ", np.shape(X_test), " #############"
	tfidf_test = get_variable(odir,"tfidf_test", lambda:get_tfidf(X_test, idf_train))

	generate_category(odir, tfidf_train, tfidf_test, Y_train)


def get_variable(odir, varname, generator):
	filename = odir+varname+".pkl"
	
	if os.path.exists(filename):
		f = open(filename, 'rb')
		var = pickle.load(f)
		print "read" , varname , 'from' , odir
	else:
		var = generator()
		f = open(filename, 'wb')
		pickle.dump(var,f)
		print "wrote" , varname , 'to' , odir
		
	f.close()
	return var


def get_idf(X):		
	no_of_docs =  np.shape(X)[0]
	unit_vector = np.matrix([[1] * no_of_docs])
	df = unit_vector * X
	idf = map(lambda i: 0 if i ==0 else np.log(no_of_docs) - np.log(i), np.matrix.flatten(np.array(df)))
	del df,unit_vector 
	return idf	
	
	
def get_tfidf(X, idf): 
	idf_matrix = np.matrix(idf)
	#p = Pool(4)
	X = map(lambda x: csr_matrix.multiply(x,idf_matrix), X)
	return X
	
	
def generate_category(odir, tfidf_train, tfidf_test, Y_train):
	cosine_function = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
	
	f = open(odir+"output.csv",'w')
	f.write('Id,Predicted\n')
	
	doc_no = 1
	for testVector in tfidf_test:
		cosine = []
		cat_list = []
		final_cat=""
		for trainVector in tfidf_train:
			cosine.append(cosine_function(trainVector.todense(), testVector.todense()))
		for x in sorted(xrange(len(cosine)), key=lambda x: cosine[x], reverse=True)[:5]:
			cat_list += Y_train[x]
		cat_list = collections.Counter(cat_list)
		for letter, count in cat_list.most_common(3):
			final_cat = final_cat + str(int(letter)) + " "
		final_cat = str(doc_no) + ',' + final_cat + '\n'
		f.write(final_cat)
		doc_no+=1
		if doc_no%50000 == 0:
			print i
 

if __name__ == "__main__":
	main(sys.argv[1:])

