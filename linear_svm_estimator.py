from sklearn.metrics import roc_curve, auc,roc_auc_score,make_scorer,accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
def linear_svm_estimator(trainX,trainY,testX,testY):
    def tuning_linear_svm(trainX,trainY):
        y_train=trainY.as_matrix()
        y_train = label_binarize(y_train.T, classes=['no','yes'])
        scores = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        svm_param ={'penalty':['l1','l2'],'C': [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e1,5e1,1e2,5e2,1e3,5e3],'class_weight':['balanced',None]} 
        svc=GridSearchCV(LinearSVC(dual=False),svm_param, cv=20, verbose=1, scoring=scores,refit='Accuracy',n_jobs=-1,return_train_score=True)
        print('# Tuning hyper-parameters for auc')
        svc.fit(trainX, y_train.reshape(y_train.shape[0],))
        print("Best parameters set found on development set:")
        print(svc.best_params_)
        return svc.best_params_
    params=tuning_linear_svm(trainX,trainY)
    svc_linear=LinearSVC(dual=False)
    svc_linear.set_params(**params)
    svc_linear.fit(trainX,trainY)
    y_test = label_binarize(testY, classes=['no','yes'])
    y_pred=svc.decision_function(testX)
    roc_area=roc_auc_score(y_test,y_pred)
    accuracy=svc_linear.score(testX,testY)
    print('AUC of linear SVM is:'+str(roc_area))
    print('Accuracy of linear SVM is:'+str(accuracy))
    return svc_linear,y_test,y_pred  # classifier, y_test and y_pred needed for roc_curve func