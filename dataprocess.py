import matplotlib.pyplot as plt
import re

epoch_num = 0
Loss = []
True_Loss = []
Accuracy = []

with open('experimentonfashionmnist.txt', 'r') as f:
    for line in f:
        text = line.strip('\n').split(',')[0]
        num = float(re.findall(r'-?\d+\.?\d*e?-?\d*?', text)[-1])
        if 'Epoch' not in text:
            if 'True Loss' in text:
                True_Loss.append(num)
            elif 'Loss' in text:
                epoch_num += 1
                Loss.append(num)
            else:
                Accuracy.append(num)

x1 = range(0, epoch_num)
x2 = range(0, epoch_num)
y1 = Accuracy
y2 = Loss
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Indicators vs. Epoches')
plt.ylabel('Validation accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.show()



