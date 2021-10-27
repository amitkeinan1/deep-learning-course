import torch.optim as optim
from matplotlib import pyplot as plt


def get_loss(model, data_loader, criterion):
    loss = 0.0
    for data_item in data_loader:
        inputs, labels = data_item
        outputs = model(inputs)
        loss += criterion(outputs, labels)
    loss_per_item = loss / len(data_loader)
    return loss_per_item

def plot_losses(train_losses, test_losses):
    pass


def train(model, train_loader, test_loader, epochs_num, criterion, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_losses = []
    test_losses = []
    epochs_count = 0

    for epoch in range(epochs_num):

        running_loss = 0.0
        for i, data_item in enumerate(train_loader, 0):
            inputs, labels = data_item

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            if epochs_count % 10000 == 0:
                train_loss = get_loss(model, train_loader, criterion)
                test_loss = get_loss(model, test_loader, criterion)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print(f"train loss: {train_loss}")
                print(f"test loss: {test_loss}")

            epochs_count += 1

    plt.plot(train_losses, 's--')
    plt.plot(test_losses, 's--')
    plt.show()

    print('Finished Training')

