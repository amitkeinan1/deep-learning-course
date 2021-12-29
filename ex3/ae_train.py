import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from ae_models import *
import numpy as np
import datetime
import os


def train_AE_model(latent_dim=64,
                   output_dir='',
                   num_epochs=20,
                   batch_size=32,
                   reload_model=False,
                   learning_rate=0.001,
                   test_interval=15,
                   load_model_path='',
                   Best=False,
                   ):
    model_name = 'AE_model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    # train_dataset = list(train_dataset)[:4096]
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8)

    if len(output_dir):
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H%M")
        final_name = '_'.join((model_name, f"latent_dim_{latent_dim}", f"lr_{learning_rate}",
                               f"BatchSize_{batch_size}", f"Epochs_{num_epochs}", time_stamp))
        model_path = os.path.join(output_dir, final_name)
        os.makedirs(model_path, exist_ok=True)

    model = Autoencoder(latent_dim=latent_dim).to(device)
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--

    outputs = []
    best_test_loss = np.inf
    train_loss_list = []
    test_loss_list = []
    itrrations = []
    for epoch in range(num_epochs):
        itr = 0
        for data in train_loader:
            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                data = next(iter(test_loader))  # get a test batch
            else:
                test_iter = False
            img, _ = data
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)
            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = loss
            else:
                test_loss = loss
            if test_iter:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{int(len(train_dataset) / float(train_loader.batch_size)) + 1}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}"
                )

        epoch_test_loss = []
        for data in test_loader:
            img, _ = data
            img = img.to(device)
            recon = model(img)
            cur_test_loss = criterion(recon, img)
            epoch_test_loss.append(cur_test_loss.cpu().detach().numpy())
        epoch_train_loss = []
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            recon = model(img)
            cur_train_loss = criterion(recon, img)
            epoch_train_loss.append(cur_train_loss.cpu().detach().numpy())
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],"
            f"Train Loss: {float(np.mean(epoch_train_loss)):.4f}, "
            f"Test Loss: {float(np.mean(epoch_test_loss)):.4f}"
        )
        train_loss_list.append(train_loss)
        if best_test_loss > np.mean(epoch_test_loss):
            best_test_loss = np.mean(epoch_test_loss)
            print("got better loss: ", best_test_loss)
            if len(output_dir):
                torch.save(model.state_dict(), os.path.join(model_path, "best_" + model.name() + ".pth"))
        test_loss_list.append(np.mean(epoch_test_loss))
        itrrations.append(epoch)
        outputs.append((epoch, float(np.mean(epoch_test_loss)), latent_dim), )
        imgs = img.cpu().detach().numpy()
        recon = recon.cpu().detach().numpy()
        if epoch % 3 == 0:
            cur_plot = plt.figure(figsize=(9, 2))
            for i, item in enumerate(imgs):
                if i >= 9: break
                plt.subplot(2, 9, i + 1)
                plt.imshow(item[0])

            for i, item in enumerate(recon):
                if i >= 9: break
                plt.subplot(2, 9, 9 + i + 1)
                plt.imshow(item[0])
            if len(output_dir):
                cur_plot.savefig(os.path.join(model_path, model.name() + f"_GtVsPredicted_epoch : {epoch}" + ".png"))
            else:
                plt.show()
            plt.close(cur_plot)

    figure = plot_losses(train_loss_list, test_loss_list, itrrations, 'TrainAndTest_Loss')
    if len(output_dir):
        figure.savefig(os.path.join(model_path, model.name() + "_TrainAndTest_Loss_Plot" + ".png"))
        plt.close(figure)
    else:
        plt.show()
        plt.close(figure)

    max_epochs = num_epochs
    return outputs, best_test_loss, model


def Q1_test_latent_dim_result(model_name='',
                              latent_dims=[2, 4, 8],  # ,16,32
                              output_dir='',
                              num_epochs=25,
                              batch_size=64,
                              reload_model=False,
                              learning_rate=0.001,
                              test_interval=100,
                              load_model_paths='',
                              Best=False):
    results = []
    if reload_model:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
        test_dataset = datasets.MNIST(
            root="~/torch_datasets", train=False, transform=transform, download=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=8
        )
        latent_dims = []
        if len(load_model_paths):
            for path in load_model_paths:
                str_latent_dims = path.split('latent_dim_')
                latent_dim = int(str_latent_dims[1].split('_')[0])
                print(latent_dim)
                latent_dims.append(latent_dim)
                print("Reloading model")
                tmp_model_name = "best_" + "Autoencoder"
                model = Autoencoder(latent_dim=latent_dim).to(device)
                model.load_state_dict(torch.load(os.path.join(path, tmp_model_name + ".pth"), device))
                epoch_test_loss = []
                criterion = nn.MSELoss()
                for data in test_loader:
                    img, _ = data
                    img = img.to(device)
                    recon = model(img)
                    cur_test_loss = criterion(recon, img)
                    epoch_test_loss.append(cur_test_loss.cpu().detach().numpy())
                results.append(np.mean(epoch_test_loss))
    else:

        for latent_dim in latent_dims:
            outputs, best_test_loss = train_AE_model(latent_dim=latent_dim, output_dir=output_dir,
                                                     num_epochs=num_epochs,
                                                     batch_size=batch_size,
                                                     reload_model=False,
                                                     learning_rate=0.001,
                                                     test_interval=100,
                                                     load_model_path='',
                                                     Best=False)
            torch.cuda.empty_cache()
            results.append(best_test_loss)
    plt.close()
    figur = plot_dim_vs_losses(latent_dims, results, 'EA Models')
    figur.savefig(os.path.join(output_dir, "compare MSE Loss With Dim Latent Space" + ".png"))
    plt.close(figur)


def Q2_test_Decorrelation_result(model_name='',
                                 latent_dims=[2, 4, 8, 16, 32],
                                 output_dir='',
                                 num_epochs=25,
                                 batch_size=64,
                                 reload_model=False,
                                 learning_rate=0.001,
                                 test_interval=100,
                                 load_model_paths='',
                                 Best=False):
    results = []
    if reload_model:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
        test_dataset = datasets.MNIST(
            root="~/torch_datasets", train=False, transform=transform, download=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=8
        )
        latent_dims = []
        if len(load_model_paths):
            for path in load_model_paths:
                str_latent_dims = path.split('latent_dim_')
                latent_dim = int(str_latent_dims[1].split('_')[0])
                print(latent_dim)
                latent_dims.append(latent_dim)
                print("Reloading model")
                tmp_model_name = "best_" + "Autoencoder"
                model = Autoencoder(latent_dim=latent_dim).to(device)
                model.load_state_dict(torch.load(os.path.join(path, tmp_model_name + ".pth"), device))
                epoch_test_loss = torch.empty((0, latent_dim))
                # print(epoch_test_loss.shape)
                for data in test_loader:
                    img, _ = data
                    img = img.to(device)
                    latent_space = model.encoder(img)
                    # print(latent_space.cpu().detach().numpy().shape)
                    torch.FloatTensor(latent_space.cpu().detach().numpy())
                    epoch_test_loss = torch.cat((epoch_test_loss, latent_space.cpu().detach()))
                    # print(epoch_test_loss.shape)
                x = torch.FloatTensor(epoch_test_loss)
                numpay_results = x.numpy()
                coaf = np.corrcoef(numpay_results.T)
                print(numpay_results.T.shape)
                # print(coaf)
                R, C = np.triu_indices(coaf.shape[0], 1)
                out = np.abs(np.einsum('ij,ij->j', coaf[:, R], coaf[:, C]))
                print(out)
                print(out.mean())
                results.append(out.mean())

    figur = plot_dim_vs_losses(latent_dims, results, train_name='Pearson cross correlation')
    figur.savefig(os.path.join(output_dir, "compare Pearson cross correlation With Dim Latent Space" + ".png"))
    plt.close(figur)
    pass


def Q3_TrainClassifier(model_name='',
                       latent_dims=12,
                       last_filter=64,
                       output_dir='',
                       num_epochs=25,
                       batch_size=64,
                       reload_model=False,
                       learning_rate=0.001,
                       test_interval=100,
                       load_model_paths='',
                       transfer_learning=False,
                       percent_of_data=0.1
                       ):
    if transfer_learning:
        if len(load_model_paths):
            str_latent_dims = load_model_paths.split('latent_dim_')
            latent_dim = int(str_latent_dims[1].split('_')[0])
            print(latent_dim)
            latent_dims.append(latent_dim)
            print("Reloading model")
            tmp_model_name = "best_" + "Autoencoder"
            model = Autoencoder(latent_dim=latent_dim).to(device)
            model.load_state_dict(torch.load(os.path.join(load_model_paths, tmp_model_name + ".pth"), device))
            encoder = model.encoder
            classifier_model = Classifier(Model_encoder=encoder, latent_dim=latent_dims, transfer_learning=True)
        else:
            print("Need to provide location for Loading module")

    else:
        classifier_model = Classifier(Model_encoder=None, latent_dim=latent_dims, transfer_learning=False,
                                      last_filter=last_filter)

    model_name = 'AE_model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    train_dataset = list(train_dataset)[:int(list(train_dataset) * percent_of_data)]
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8)


def plot_losses(train_losses, test_losses, batch_counts, train_name):
    figure = plt.figure()
    plt.plot(batch_counts, train_losses, 'o--', label="train", alpha=0.8)
    plt.plot(batch_counts, test_losses, 'o--', label="test", alpha=0.8)

    plt.title(f"train and test losses - {train_name}")
    plt.xlabel("batches count")
    plt.ylabel("loss per item")
    plt.legend()
    # plt.savefig(os.path.join(FIGS_PATH, str(random.random()) + ".png"))
    # plt.show()
    return figure


def plot_dim_vs_losses(train_losses, test_losses, train_name):
    figure = plt.figure()
    plt.plot(train_losses, test_losses, label="test", alpha=0.8)

    plt.title(f"test losses - {train_name}")
    plt.xlabel("latent dim count")
    plt.ylabel("loss per item")
    plt.legend()
    # plt.savefig(os.path.join(FIGS_PATH, str(random.random()) + ".png"))
    # plt.show()
    return figure


if __name__ == '__main__':
    output_dir = "/home/eldan/Uni/EaModels"
    model_reload = ["/home/eldan/Uni/EaModels/AE_model_latent_dim_2_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1805",
                    "/home/eldan/Uni/EaModels/AE_model_latent_dim_4_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1806",
                    "/home/eldan/Uni/EaModels/AE_model_latent_dim_4_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1806",
                    "/home/eldan/Uni/EaModels/AE_model_latent_dim_8_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1807",
                    "/home/eldan/Uni/EaModels/AE_model_latent_dim_16_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1808",
                    ]
    '''"/home/eldan/Uni/EaModels/AE_model_latent_dim_16_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1714",
                    "/home/eldan/Uni/EaModels/AE_model_latent_dim_32_lr_0.001_BatchSize_64_Epochs_10_2021-12-25_1715"'''
    # Q1_test_latent_dim_result(output_dir =output_dir, num_epochs = 10,load_model_paths = model_reload, reload_model=True)
    Q2_test_Decorrelation_result(output_dir=output_dir, num_epochs=10, load_model_paths=model_reload, reload_model=True)
    # print(outputs)
