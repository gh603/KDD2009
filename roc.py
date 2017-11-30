from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
# choose one classifier to run
classifier = GaussianNB()
y_score = classifier.fit(trainX, trainY).predict_proba(testX)
#print(testY.as_matrix())
#print(y_score[:,1])

y_test=testY.as_matrix()
y_test[y_test=='no']=0
y_test[y_test=='yes']=1
y_test=y_test.T
#print(y_test.shape)

def draw_roc(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

        
draw_roc(y_test, y_score)