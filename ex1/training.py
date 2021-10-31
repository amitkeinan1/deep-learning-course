import random
import torch.optim as optim
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

FIGS_PATH = r"C:\Users\AMIT\Google Drive\cloud\אקדמיה\נוכחיים\DL\deep-learning-course\ex1\figs"


def get_loss(model, data_loader, criterion):
    loss = 0.0
    for batch in data_loader:
        inputs, labels = batch
        outputs = model(inputs)
        loss += criterion(outputs, labels)
    loss_per_item = loss / len(data_loader)
    return loss_per_item


def plot_losses(train_losses, test_losses, batch_counts, train_name):
    plt.plot(batch_counts, train_losses, 'o--', label="train", alpha=0.8)
    plt.plot(batch_counts, test_losses, 'o--', label="test", alpha=0.8)

    plt.title(f"train and test losses - {train_name}")
    plt.xlabel("batches count")
    plt.ylabel("loss per item")
    plt.legend()
    plt.savefig(os.path.join(FIGS_PATH, str(random.random()) + ".png"))
    plt.show()


def train(model, train_loader, test_loader, epochs_num, criterion, lr, train_name):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_losses = []
    test_losses = []
    batch_counts = []

    total_batch = 0

    for epoch in tqdm(range(epochs_num)):
        print(f"epoch {epoch}")

        running_loss = 0.0
        for batch, data_item in enumerate(train_loader, 0):
            inputs, labels = data_item

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch % 1000 == 999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch + 1, running_loss / 2000))
                running_loss = 0.0

            if total_batch % 200 == 199:
                train_loss = get_loss(model, train_loader, criterion)
                test_loss = get_loss(model, test_loader, criterion)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                batch_counts.append(total_batch)
                print(f"train loss: {train_loss}")
                print(f"test loss: {test_loss}")

            total_batch += 1

    plot_losses(train_losses, test_losses, batch_counts, train_name)

    print('Finished Training')
