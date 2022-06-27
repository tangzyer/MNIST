import matplotlib.pyplot as plt
import sys
import re

epoch_num = 0
Loss = []
True_Loss = []
Accuracy = []

with open('./experimentmlp2.txt', 'r') as f:
    for line in f:
        text = line.strip('\n').split(',')[0]
        num = float(re.findall(r'-?\d+\.?\d*e?-?\d*?', text)[-1])
        if 'True Loss' in text:
            epoch_num += 1
            True_Loss.append(num)
        elif 'Loss' in text:
            Loss.append(num)
        else:
            Accuracy.append(num)

x1 = range(0, epoch_num)
x2 = range(0, epoch_num)
x3 = range(0, epoch_num)
y1 = Accuracy
y2 = Loss
y3 = True_Loss
plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Validation indicators vs. epoches')
plt.ylabel('Validation accuracy')
plt.subplot(3, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.subplot(3, 1, 3)
plt.plot(x3, y3, '.-')
plt.xlabel('epochs')
plt.ylabel('True loss')
# plt.savefig('Validation indicators '+mode+".png")
plt.show()



