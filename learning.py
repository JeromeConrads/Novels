import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit,RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
import scipy as sp
import scipy.stats
from sklearn.multiclass import OneVsRestClassifier as ORC

from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

from scipy.stats import randint
from collections import Counter
from scipy.stats import uniform


def main():
    X = np.loadtxt("./data/X50-1.csv").T
    y = np.loadtxt("./data/y50-1.csv").T



    # for rm in [3,4]:
    #     X = X[y != rm]
    #     y = y[y != rm]



   

    #X = X.T
    
    #XXX was ist dummy classifier/was bedeuten diese zahlen
    print(X.shape, y.shape)
    dc = DummyClassifier(strategy = "most_frequent")
    dc.fit(X,y)


    print("Most Frequent", dc.score(X,y))
    dc = DummyClassifier(strategy = "stratified") ; dc.fit(X,y)
    print("Stratified", dc.score(X,y))
    #clf = LogisticRegression(class_weight="auto")
    #clf = RidgeClassifierCV(class_weight="auto", normalize=False)
    # #model =
    #model = clf.fit(X, y)


    #print confusion_matrix(y,dc.predict(X))
    #exit()



    # check the accuracy on the training set
    #score = model.score(X, y)
    #print score 
    #clf=ExtraTreesClassifier(n_estimators=1500,n_jobs=2 ,random_state=10,max_depth=5)
    #clf=ExtraTreesClassifier(n_estimators=150,n_jobs=2 ,random_state=10,min_samples_split= 5, min_samples_leaf= 1, max_features= "sqrt")
    clf=ExtraTreesClassifier(n_estimators=500,bootstrap=True,oob_score=True,class_weight= "balanced_subsample",ccp_alpha=0.0005,max_depth=15)
    #clf = ExtraTreesClassifier(n_estimators=3500, n_jobs = 2, random_state=10)
    #clf = ORC(clf)
    #clf = RandomForestClassifier(n_estimators=1000, min_samples_split=100, n_jobs = 2, random_state = 10)
    

   # clf = DecisionTreeClassifier(max_depth= 20)
    #clf = GradientBoostingClassifier(n_estimators= 30, min_samples_split=5, verbose=True, learning_rate=0.8, subsample=0.5)
    #clf = NuSVC()
    #clf = SVC(class_weight="auto")
    #clf = Pipeline([('anova', preprocessing.StandardScaler()), ('svc', SGDClassifier(shuffle=True, class_weight="auto", n_iter=20, l1_ratio = 1))])
    #clf = Pipeline([('anova', preprocessing.StandardScaler()), ('svc', SVC(kernel="poly", C = 2.0))])
    #clf = Pipeline([('anova', preprocessing.StandardScaler()), ('svc', NuSVC(nu=0.9))])

    hyper_param ={
    'n_estimators': [500],
    "ccp_alpha": uniform(0.0005, 0.002 - 0.0001),
    "max_depth": [15]
    }
    #clf= RandomizedSearchCV(ExtraTreesClassifier(random_state=10), hyper_param, cv=5, random_state=10,verbose=3)  
    #clf.fit(X, y)
    #print(clf.best_params_)
    #exit()
    
    
    
    ss = StratifiedShuffleSplit(n_splits=10, random_state=23)
    scores = []
    cms = []
    cms_train = []
    for i, (train_index, test_index) in enumerate(ss.split(X,y)):
        print("Shuffle %d"%(i,),)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        #smote = SMOTE(random_state=0)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        
        #sample_dict = { n : 25 for n in range(6) }
        #ros = RandomUnderSampler(random_state=0,sampling_strategy= sample_dict)
        #X_train, y_train = ros.fit_resample(X_train, y_train)
        #samplepipe = make_pipeline(RandomUnderSampler(random_state=0), SMOTE(random_state=0))
        #X_train, y_train = samplepipe.fit_resample(X_train, y_train)


        counts = Counter(y_train)
        # Print the count for each Genre
        for key, value in counts.items():
            print(f"Genre {key} is represented {value} times in train set.")
            

            
        #print("%s %s" % (train_index, test_index))
        clf.fit(X_train, y_train)
        y_hat_train = clf.predict(X_train)
        y_hat = clf.predict(X_test)
        
        #score = clf.score(X[test_index], y[test_index])
        score_test = accuracy_score(y_test, y_hat)
        score_train= accuracy_score(y_train, y_hat_train)
        precision = precision_score(y_test, y_hat, average='macro')
        precision2 = precision_score(y_test, y_hat, average='micro')
        recall = recall_score(y_test, y_hat, average='macro')
        recall2 = recall_score(y_test, y_hat, average='micro')
        f1 = f1_score(y_test, y_hat, average='macro')   
        f12 = f1_score(y_test, y_hat, average='micro')
        print("train accuracy: ",score_train)     
        print("test accuracy: ",score_test)
        print("Macro precision: ",precision)        
        print("Micro precision: ",precision2)
        print("Macro recall: ",recall)
        print("Micro recall: ",recall2)
        print("Macro f1: ",f1)
        print("Micro f1: ",f12)
        
        cm = confusion_matrix(y_test, y_hat,normalize = "true")
        cm_train =confusion_matrix(y_train, y_hat_train,normalize = "true")
        #print m.type()
        scores.append(score_test)
        #print score, np.array(scores).mean()
        #print(cm,type(cm))
        #cm = np.array(cm,dtype="float")
        print("test:")
        print(cm)
        print("train;")
        print(cm_train)
        #exit()
        #normiert
        #cm =  cm/cm.sum(axis = 1)
        #print cm.shape
        #exit()
        #print cm
        cms.append(cm)
        cms_train.append(cm_train)

    scores = np.array(scores)
    print("ERF", scores.mean())
    #exit()
    #print cms
    cms = np.array(cms)
    #print cms.shape
    cms = np.mean(cms, axis = 0)
    print("final Confusion matrix:")
    print(cms)
    plt.matshow(cms)
    #plt.title('Confusion matrix')
    plt.colorbar(fraction=0.046, pad=0.2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



    labels = ["fantasy".upper(), "humor".upper(), "science fiction".upper(), "horror".upper(), "mystery".upper(), "western".upper()]
    plt.xticks(range(6), labels )
    plt.yticks(range(6), labels)
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    mean =scores.mean()
    plt.text(0, -0.1, f"Mean Accuracy: {mean}", fontsize=12,
         horizontalalignment="left", verticalalignment="center", transform=plt.gca().transAxes)
    plt.savefig("./data/confusion100.eps", transparent=False, bbox_inches="tight")
    plt.show()
        #print m


    clf.fit(X,y)
    importances = clf.feature_importances_
    ft = np.array([tree.feature_importances_ for tree in clf.estimators_])
    std = np.std(ft, axis=0)
    print(std.shape)


    indices = np.argsort(importances)[::-1]

    sem1 =  std / np.sqrt(ft.shape[0])
    cf = sem1*1.96

    print("Feature ranking:")

    for f in range(100):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        print(f+1," feature ", indices[f]," (",importances[indices[f]],")")

    # Plot the feature importances of the forest
    plt.figure()
    #plt.title("Feature importances")
    print(importances[indices].shape)
    important = 30 
    plt.barh(range(important), width = importances[indices][:important][::-1],
           color="c", xerr=cf[indices][:important][::-1], height = 0.5,  align="center", ecolor='r')

    feature_names = ["ANGER", "DISGUST", "FEAR", "JOY", "SADNESS", "SURPRISE"]

    yticks = []
    per_feature = 50
    print(std[indices][:important])
    print(importances[indices][:important])
    print(cf)
    for i in indices:
        yticks.append(feature_names[int(i/per_feature)] + " " + str(int(i%per_feature)))
    #yticks[::-1]
    #plt.tick_params(axis='x', labelsize=5)


    plt.yticks(range(120)[::-1], yticks)
    plt.ylim([-1, important])
    plt.savefig("./data/importances100.eps", bbox_inches='tight')
    print(len(indices))
    plt.show()


if __name__ == '__main__':
    main()




