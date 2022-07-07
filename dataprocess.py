import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

epoch_num = 0
Loss = []
True_Loss = []
Accuracy = []


lda_gmm = [[],[],[],[],[]]
lda_mom = [[],[],[],[],[]]

with open ('lda_mnist.txt', 'r') as f:
    for line in f:
        text = line.strip('\n').split(',')[0]
        nums = re.findall(r'-?\d+\.?\d*e?-?\d*?', text)
        n = int(float(nums[-1]))
        acc = float(nums[-2])
        if 'GMM' in text:
            i = n // 10000
            lda_gmm[i].append(acc)
        else:
            i = n // 10000
            lda_mom[i].append(acc)
    
lda_gmm_avg = np.mean(np.array(lda_gmm), axis=1)
lda_gmm_std = np.log(np.std(np.array(lda_gmm), axis=1))
lda_mom_avg = np.mean(np.array(lda_mom), axis=1)
lda_mom_std = np.log(np.std(np.array(lda_mom), axis=1))

df_nn = pd.read_csv('06290918nn.csv')
nn_avg = df_nn['acc_nn_avg'].values
nn_std = df_nn['acc_nn_std'].values

df_knn = pd.read_csv('06290918knn.csv')
knn_avg = df_knn['acc_knn_avg'].values
knn_std = df_knn['acc_knn_std'].values

x1 = [0.1,0.3,0.5,0.7,0.9]
plt.cla()
plt.subplot(2, 1, 1)
plt.ylabel('average accuracy')
plt.plot(x1, nn_avg, '.-')
plt.plot(x1, knn_avg, ',-')
plt.plot(x1, lda_gmm_avg, 'o-')
plt.plot(x1, lda_mom_avg, 'v-')
plt.legend(['NN', r'$k$'+'-NN', 'LDA_GMM', 'LDA_MoM'])
plt.subplot(2, 1, 2)
plt.ylabel(r"$\ln$"+'(accuracy std)')
plt.xlabel('data size '+r'$\alpha=n/N$')
plt.plot(x1, nn_std, '.-')
plt.plot(x1, knn_std, ',-')
plt.plot(x1, lda_gmm_std, 'o-')
plt.plot(x1, lda_mom_std, '^-')
plt.legend(['NN', r'$k$'+'-NN', 'LDA_GMM', 'LDA_MoM'])
plt.savefig('mnistdata.png')
# plt.legend(['softmax', 'knn', 'LDA_GMM', 'LDA_MoM', 'Bayesian'])
# plt.savefig('withsharedcov_3'+ ".png")
# df = {'n':iters, 'knn': acc_knn, 'LDA_GMM': acc_lda_gmm, 'bayesian': acc_bayes, 'LDA_MoM': acc_lda_mom, 'Softmax': acc_softmax}
# df = pd.DataFrame(df)
# df.to_csv('log_csv_4.csv')


# with open('experimentonfashionmnist.txt', 'r') as f:
#     for line in f:
#         text = line.strip('\n').split(',')[0]
#         num = float(re.findall(r'-?\d+\.?\d*e?-?\d*?', text)[-1])
#         if 'Epoch' not in text:
#             if 'True Loss' in text:
#                 True_Loss.append(num)
#             elif 'Loss' in text:
#                 epoch_num += 1
#                 Loss.append(num)
#             else:
#                 Accuracy.append(num)

# x1 = range(0, epoch_num)
# x2 = range(0, epoch_num)
# y1 = Accuracy
# y2 = Loss
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Indicators vs. Epoches')
# plt.ylabel('Validation accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('epochs')
# plt.ylabel('Train loss')
# plt.show()



