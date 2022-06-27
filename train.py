from torchvision import models
import torch
import matplotlib.pyplot as plt

epochs = 128
learning_rate = 0.001

def train(train_loader, X_validation, model):
    criterion = LogEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
    # weight_decay=1e-4)
    train_loss = []
    validation_accuracy = []
    for epoch in range(epochs):
        sum_loss = 0.0
        for data in train_loader:
            # data_size = data[0].shape[0]
            # use_cuda = torch.cuda.is_available()
            img, label = data
            img = img.reshape(-1, 3, 32, 32)
            # label = label.reshape(2, -1, 4)
            # if use_cuda:
            #     img, label = img.cuda(), label.cuda()
            # imgs, labels_a, labels_b, lam = mixup_data(img, label,
            #                                            0.75, use_cuda)
            # imgs, labels_a, labels_b = map(Variable, (imgs, labels_a, labels_b))
            # outputs = net(inputs)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            # loss = mixup_criterion(criterion, out, labels_a, labels_b, lam)
            loss = criterion(out, label)
            print_loss = loss.data.item()
            # print('loss',print_loss)
            sum_loss += print_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(sum_loss)
        # 监视验证集
        with torch.no_grad():
            val_input = torch.from_numpy(np.array(X_validation)).to(torch.float32)
            val_inputs = val_input.reshape(-1, 3, 32, 32)
            if torch.cuda.is_available():
                val_inputs = Variable(val_inputs).cuda()
            val_outputs = model(val_inputs)
            print('Epoch:', epoch, 'output')
            # print(val_outputs)
            # print(validation_labels)
            val_preds = np.argmax(val_outputs.cpu().detach().numpy(), axis=1)
            acc = accuracy_score(val_preds, validation_labels)
            print('Epoch:', epoch, 'Val Acc:', acc)
            validation_accuracy.append(acc)
        print('Epoch', epoch, 'Loss:', sum_loss)
    epoch_num = epochs
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    y1 = validation_accuracy
    y2 = train_loss
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Validation indicators vs. epoches')
    plt.ylabel('Validation accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    # plt.savefig('Validation indicators '+mode+".png")
    plt.show()
    return model