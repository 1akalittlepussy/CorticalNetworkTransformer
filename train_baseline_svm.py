import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import KFold
from utils import *

###Parameters###
proportion = 0.20
atlas = ''
node_number = 90
no_folds = 10
iter_time = 10
#V, A, labels = preprocess('/home/wwh/Project/data/multi-site')
V, A, labels = preprocess('F:/graphmix-master-new/codes_graph/HCP62ROI')
A_shape = A.shape  #
data_size = A_shape[0]
###model###
#model = svm.SVC(C=10, kernel='sigmoid')#
model = RandomForestClassifier(random_state=125)  # n_estimators=5,random_state=125
# model = MLPClassifier(solver='lbfgs', activation='tanh',alpha=0.0001, hidden_layer_sizes=(32,32), random_state=125)
# random_s = np.array([0,20, 40, 60, 80, 100, 120, 140, 160, 180], dtype=int)
# random_s = np.array([125, 125, 125, 125, 125, 125, 125, 125, 125, 125], dtype=int)
random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
acc_set = np.zeros((iter_time, 1))
std_set = np.zeros((iter_time, 1))
sen_set = np.zeros((iter_time, 1))
sen_std_set = np.zeros((iter_time, 1))
spe_set = np.zeros((iter_time, 1))
spe_std_set = np.zeros((iter_time, 1))
f1_set = np.zeros((iter_time, 1))
f1_std_set = np.zeros((iter_time, 1))
auc_set = np.zeros((iter_time, 1))
auc_std_set = np.zeros((iter_time, 1))
for iter_num in range(iter_time):
    acc = []
    sen = []
    spe = []
    f1_score = []
    auc_score = []
    for i in range(no_folds):
        inst = KFold(n_splits=no_folds, shuffle=True, random_state=random_s[iter_num])
        fold_id = i
        KFolds = list(inst.split(np.arange(data_size)))
        train_idx, test_idx = KFolds[fold_id]
        no_samples_train = train_idx.shape[0]
        no_samples_test = test_idx.shape[0]
        train_x = A[train_idx, ...]
        test_x = A[test_idx, ...]
        train_y = labels[train_idx, ...]
        test_y = labels[test_idx, ...]
        print('Data ready. no_samples_train:', no_samples_train, 'no_samples_test:', no_samples_test)
        model.fit(train_x, train_y)
        pre = model.predict(test_x)
        accuracy, sensitivity, specificity, f1, auc = compute_metrics(test_y, pre)
        print('Fold %d max accuracy: %g  sensitivity: %g specificity: %g f1_score: %g auc: %g' % (
            i, accuracy, sensitivity, specificity, f1, auc))
        acc.append(accuracy)
        sen.append(sensitivity)
        spe.append(specificity)
        f1_score.append(f1)
        auc_score.append(auc)
        verify_dir_exists('results/')
        with open('results/%s.txt' % 'svm_every_fold', 'a+') as file:
            file.write('%s\t%d-fold\t%.4f sensitivity\t%.4f specificity\t%.4f f1_score\t%.4f auc\t%.4f \n' % (
                str(datetime.now()), i, accuracy, sensitivity, specificity, f1, auc))
    mean_acc, std_acc, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, mean_f1_score, std_f1_score, mean_auc, std_auc = mean_metrics(
        acc, sen, spe, f1_score, auc_score)
    print_each_iter(no_folds, mean_acc, std_acc, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity,
                    mean_f1_score, std_f1_score, mean_auc, std_auc)

    acc_set[iter_num] = mean_acc
    std_set[iter_num] = std_acc
    sen_set[iter_num] = mean_sensitivity
    sen_std_set[iter_num] = std_sensitivity
    spe_set[iter_num] = mean_specificity
    spe_std_set[iter_num] = std_specificity
    f1_set[iter_num] = mean_f1_score
    f1_std_set[iter_num] = std_f1_score
    auc_set[iter_num] = mean_auc
    auc_std_set[iter_num] = std_auc

print_final(proportion, iter_time, acc_set, std_set, sen_set, sen_std_set, spe_set, spe_std_set, f1_set, f1_std_set,
            auc_set, auc_std_set)