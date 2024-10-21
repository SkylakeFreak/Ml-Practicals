from sklearn import datasets

dataset=datasets.load_iris()
x=dataset.data
y=dataset.target


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print("Accuracy is: ",accuracy)
print("Confusion Matrix is:",confusion_matrix)

labels=dataset.target_names

import seaborn as sns

sns.heatmap(confusion_matrix,annot=True,fmt="d",xticklabels=labels,cmap="Blues",cbar=False)

import matplotlib.pyplot as plt

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
