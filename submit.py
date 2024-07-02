import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
import time as tm

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
# # ################################
# # #  Non Editable Region Ending  #
# # ################################
	
	X_train_final=my_map(X_train)

	#BEST RESULTS OBTAINED USING LOGISTIC REGRESSION
	C=125
	tol=0.001
	penalty='l2'
	solver='lbfgs'

	clf=LogisticRegression(C=C,tol=tol,penalty=penalty,solver=solver)
	clf.fit(X_train_final,y_train)
	w=clf.coef_
	b=clf.intercept_
	return w.T[:,0],b



################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	no_challenges=X.shape[0]
	Xk=1-2*X
	Xkk=1-2*X
	Xk[:,0]=np.prod(Xk,axis=1)
	for i in range(1,32):
		Xk[:,i]=Xk[:,i-1]/Xkk[:,i-1]

	feat=[]
	Xk = Xk.reshape(-1,32,1)
	for i in range(no_challenges):
		result = np.triu(np.dot(Xk[i], Xk[i].T), k=1).reshape(-1)
		result = np.append(result[result != 0], Xk[i])
		feat.append(result)
	feat=np.array(feat)
	return feat
