# matplotlib notebook

import numpy as np

from pandas import read_csv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.cross_validation import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt

# Load the data into Python
df = read_csv("../python-data-science-exercise/data/Default.csv", index_col=0)

# Display first few rows of our dataset
df.head()

# Check dimensionality
df.shape

# Convert strings to numerical values and reassign to 'df' variable
df = df.replace("Yes", 1)
df = df.replace("No", 0)
# df.head

# Use train_test_split to create training and test sets and assign to variables df_train and df_test
df_train, df_test = train_test_split(df, test_size=0.1)

# Define feature columns "student", "balance", and "income" as a list
feature_cols = ["student", "balance", "income"]

# Define training feature and target values
X_train = df_train[feature_cols]
Y_train = df_train.default

# Define test feature and target values
X_test = df_test[feature_cols]
Y_test = df_test.default

# Train classifier
print "Training classifier..."
gnb = GaussianNB()
clf = gnb.fit(X_train, Y_train)
print "Training done!"

# Test classifier
print "Testing classifier..."
Y_pred = clf.predict(X_test)
print Y_pred

print "Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum())

# Display confusion matrix
print confusion_matrix(Y_test, Y_pred)

# Display histogram
barwidth = 0.5
index = np.arange(len(np.unique(Y_test)))
plt.figure()
plt.bar(index, np.bincount(Y_test), barwidth)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(index + 0.5 * barwidth, ('0', '1'))
plt.show()

print classification_report(Y_test, Y_pred)

# Get sample scores for ROC curve
Y_score = clf.predict_proba(X_test)

# Compute ROC metrics
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score[:, 1], pos_label=1)
roc_auc = auc(false_positive_rate, true_positive_rate)

# Plot ROC curve
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Configure cross validation
cv = StratifiedKFold(df.default, n_folds=6)

# Create new plot
plt.figure()

# Get indices for training and test subsets for each fold
for i, (train, test) in enumerate(cv):
    # Get training data
    X_train = df[feature_cols].as_matrix()[train, :]
    Y_train = df.default.as_matrix()[train,]

    # Get test data
    X_test = df[feature_cols].as_matrix()[test, :]
    Y_test = df.default.as_matrix()[test,]

    # Configure classifier
    clf = GaussianNB()

    # Learn classification model
    clf.fit(X_train, Y_train)

    # Get sample scores for ROC curve
    Y_score = clf.predict_proba(X_test)

    # Compute ROC metrics
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score[:, 1], pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Plot ROC curve
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = %0.2f' % roc_auc)

plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
