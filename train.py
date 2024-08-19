from model import *
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# 记录数据集长度

train_data_size = len(train_data)
test_data_size = len(test_data)

print('训练集长度:{}'.format(train_data_size))
print('测试集长度:{}'.format(test_data_size))

print(train_data.data[0].shape)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)

# 模型实例化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = ResNet18(Residual).to(device)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-3  # 10^-3
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=1e-5)

# 记录训练次数
total_train_step = 0

# 记录测试次数
total_test_step = 0

# 训练的轮数
epoch = 20

writer = SummaryWriter('./logs')

best_test_loss = float('inf')

for i in range(epoch):
    print('------------------------Eopch {}/{}------------------------'.format(i+1, epoch))

    resnet.train()
    total_train_loss = 0
    for data in train_loader:
        img, labels = data
        img = img.to(device)
        labels = labels.to(device)
        output = resnet(img)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        total_train_loss += loss.item()

        if total_train_step % 100 == 0:
            print('Epoch {}/{}, Train Step: {}, Loss: {:.4f}'.format(i + 1, epoch, total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 计算平均训练损失
    avg_train_loss = total_train_loss / len(train_loader)
    writer.add_scalar('avg_train_loss', avg_train_loss, i)
    print('Epoch {}/{}, 平均训练损失: {:.4f}'.format(i + 1, epoch, avg_train_loss))


    # 测试模式
    resnet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            output = resnet(img)
            loss = loss_fn(output, labels)
            total_test_loss += loss.item()

            accuracy = (output.argmax(1) == labels).sum()
            total_accuracy += accuracy

    # 计算平均测试损失
    avg_test_loss = total_test_loss / len(test_loader)
    writer.add_scalar('avg_test_loss', avg_test_loss, total_test_step)
    total_test_step +=1
    print('Epoch {}/{}, 平均测试损失: {:.4f}'.format(i + 1, epoch, avg_test_loss))

    # 整体准确率
    print('total_accuracy:{}'.format(total_accuracy/test_data_size))
    writer.add_scalar('total_accuracy', total_accuracy/test_data_size, total_test_step)

    # 保存模型
    if avg_test_loss<best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(resnet.state_dict(), './models/best_resnet.pth')
        print('测试集上最佳模型已保存: best_resnet.pth')

writer.close()


