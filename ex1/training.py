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


def plot_losses(train_losses, test_losses, batch_counts):
    plt.plot(batch_counts, train_losses, 'o--', label="train", alpha=0.8)
    plt.plot(batch_counts, test_losses, 'o--', label="test", alpha=0.8)

    plt.title("train and test losses")
    plt.xlabel("batches count")
    plt.ylabel("loss per item")
    plt.legend()
    plt.show()


def train(model, train_loader, test_loader, epochs_num, criterion, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_losses = []
    test_losses = []
    batch_counts = []
    batches_count = 0

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

            if batches_count % 10000 == 0:
                train_loss = get_loss(model, train_loader, criterion)
                test_loss = get_loss(model, test_loader, criterion)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                batch_counts.append(batches_count)
                print(f"train loss: {train_loss}")
                print(f"test loss: {test_loss}")

            batches_count += 1

    plot_losses(train_losses, test_losses, batch_counts)

    print('Finished Training')

