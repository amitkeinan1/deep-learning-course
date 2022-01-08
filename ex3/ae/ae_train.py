import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from ae_models import *
import numpy as np
import datetime
from scipy.stats import pearsonr
import os
from ae_evaluation import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
# device = torch.device("cpu")
import copy
import csv


def train_AE_model(latent_dim=64,
                   output_dir='',
                   num_epochs=20,
                   batch_size=32,
                   learning_rate=0.001,
                   test_interval=15,
                   number_of_layer=None
                   ):
    model_name = 'AE_model'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    # train_dataset = list(train_dataset)[:4096]
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0)
    if number_of_layer is not None and number_of_layer > 1:
        model_name = f"{model_name}_with_{number_of_layer}_FC"
        model = Autoencoder_layer(latent_dim=latent_dim, number_of_layer=number_of_layer).to(device)
    else:
        model = Autoencoder(latent_dim=latent_dim).to(device)

    if len(output_dir):
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H%M")
        final_name = '_'.join((model_name, f"latent_dim_{latent_dim}", f"lr_{learning_rate}",
                               f"BatchSize_{batch_size}", f"Epochs_{num_epochs}", time_stamp))
        model_path = os.path.join(output_dir, final_name)
        os.makedirs(model_path, exist_ok=True)

    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)

    outputs = []
    best_test_loss = np.inf
    train_loss_list = []
    test_loss_list = []
    itrrations = []
    print(model)
    # print(model.encoder)
    # print(model.decoder)

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

        # run Train and Test Data set to get avreage loss on all data set at end of epoch
        epoch_test_loss = []
        for data in test_loader:
            img, _ = data
            img = img.to(device)
            recon = model(img)
            cur_test_loss = criterion(recon, img)
            epoch_test_loss.append(cur_test_loss.cpu().detach().numpy())

        average_test_loss = np.mean(epoch_test_loss)
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
        average_train_loss = np.mean(epoch_train_loss)
        train_loss_list.append(average_train_loss)
        if best_test_loss > average_test_loss:
            best_test_loss = average_test_loss
            print("got better loss: ", best_test_loss)
            if len(output_dir):
                torch.save(model.state_dict(), os.path.join(model_path, "best_" + model.name() + ".pth"))
        test_loss_list.append(np.mean(epoch_test_loss))
        itrrations.append(epoch)
        outputs.append((epoch, float(average_test_loss), latent_dim), )
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
    plt.close()
    figure = plot_losses(train_loss_list, test_loss_list, itrrations, 'TrainAndTest_Loss', use_ticks=True)
    if len(output_dir):
        figure.savefig(os.path.join(model_path, model.name() + "_TrainAndTest_Loss_Plot" + ".png"))
        plt.close(figure)
    else:
        plt.show()
        plt.close(figure)

    return outputs, best_test_loss, model


def Q1_test_latent_dim_result(model_name='',
                              latent_dims=[2, 4, 8, 10],  #
                              output_dir='',
                              num_epochs=25,
                              batch_size=64,
                              reload_model=False,
                              learning_rate=0.001,
                              test_interval=100,
                              load_model_paths='',
                              number_of_layer=None
                              ):
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
            list_of_dirs = [cur_dir for cur_dir in os.listdir(load_model_paths) if
                            os.path.isdir(os.path.join(load_model_paths, cur_dir))] if os.path.isdir(
                load_model_paths) else None
            full_paths = [os.path.join(load_model_paths, cur_model) for cur_model in
                          sorted(list_of_dirs, key=lambda dir_name: int(dir_name.split('latent_dim_')[1].split('_')[0]))
                          if
                          os.path.isdir(os.path.join(load_model_paths, cur_model))] if os.path.isdir(
                load_model_paths) else load_model_paths
            for path in full_paths:
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
            outputs, best_test_loss, model = train_AE_model(latent_dim=latent_dim,
                                                            output_dir=output_dir,
                                                            num_epochs=num_epochs,
                                                            batch_size=batch_size,
                                                            learning_rate=0.001,
                                                            test_interval=100,
                                                            number_of_layer=number_of_layer)
            torch.cuda.empty_cache()
            results.append(best_test_loss)
    plt.close()
    figur = plot_dim_vs_placeholder(latent_dims, results, 'EA Models')
    figur.savefig(os.path.join(output_dir, f"compare MSE Loss With Dim Latent Space_{number_of_layer}_layers" + ".png"))
    plt.close(figur)
    return latent_dims, results


def test_different_autoencoders_Q1(latent_dims=(2, 4, 6, 8, 10), output_dir='', layer_count=(1, 2, 4), num_epochs=20):
    layer_count = layer_count
    results = []
    latent_dims = list(latent_dims)  #
    for n in layer_count:
        cur_latent_dims, cur_results = Q1_test_latent_dim_result(output_dir=output_dir,
                                                                 num_epochs=num_epochs,
                                                                 reload_model=False,
                                                                 latent_dims=latent_dims,
                                                                 load_model_paths=output_dir,
                                                                 batch_size=128,
                                                                 number_of_layer=n)
        results.append(cur_results)
    print(layer_count)
    for i in range(len(layer_count)):
        print(results[i])
        print(latent_dims)
        plt.plot(latent_dims, results[i], 'o--', label=f"{layer_count[i]} layers", alpha=0.8)
    plt.title(f"Number Layers, latent dim Vs loss")
    plt.xlabel("latent dim")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "compare result on number of layers" + ".png"))


def Q2_test_Decorrelation_result(
        output_dir='',
        load_model_paths='',
        Train_set=False):
    results = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])

    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=Train_set, transform=transform, download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8
    )
    latent_dims = []
    if len(load_model_paths):
        list_of_dirs = [cur_dir for cur_dir in os.listdir(load_model_paths) if
                        os.path.isdir(os.path.join(load_model_paths, cur_dir))] if os.path.isdir(
            load_model_paths) else None
        full_paths = [os.path.join(load_model_paths, cur_model) for cur_model in
                      sorted(list_of_dirs, key=lambda dir_name: int(dir_name.split('latent_dim_')[1].split('_')[0])) if
                      os.path.isdir(os.path.join(load_model_paths, cur_model))] if os.path.isdir(
            load_model_paths) else load_model_paths
        for path in full_paths:
            str_latent_dims = path.split('latent_dim_')
            latent_dim = int(str_latent_dims[1].split('_')[0])
            print(latent_dim)
            latent_dims.append(latent_dim)
            print("Reloading model")
            tmp_model_name = "best_" + "Autoencoder"
            model = Autoencoder(latent_dim=latent_dim).to(device)
            model.load_state_dict(torch.load(os.path.join(path, tmp_model_name + ".pth"), device))
            epoch_test_loss = torch.empty((0, latent_dim))
            for j, data in enumerate(test_loader):
                img, _ = data
                img = img.to(device)
                latent_space = model.encoder(img)
                torch.FloatTensor(latent_space.cpu().detach().numpy())
                epoch_test_loss = torch.cat((epoch_test_loss, latent_space.cpu().detach()))
            x = torch.FloatTensor(epoch_test_loss).numpy().T
            coralations = []
            for dim_1 in range(x.shape[0]):
                for dim_2 in range(1, x.shape[0] - dim_1):
                    coralations.append(pearsonr(x[dim_1], x[dim_2 + dim_1])[0])
            print(np.abs(coralations).mean())
            print(f"number of coralations {len(coralations)}")
            results.append(np.abs(coralations).mean())
    else:
        raise "Lack of path of trained models "
    print(results)
    print(latent_dims)
    test_or_train_set = "Train set" if Train_set else "Test Set"
    figur = plot_dim_vs_placeholder(latent_dims, results,
                                    train_name='Pearson cross correlation of ' + test_or_train_set,
                                    label=test_or_train_set)
    figur.savefig(os.path.join(output_dir,
                               "compare Pearson cross correlation With Dim Latent Space of " + test_or_train_set + ".png"))
    plt.close(figur)


def TrainClassifier(latent_dim=12,
                    output_dir='',
                    num_epochs=25,
                    batch_size=64,
                    learning_rate=0.0001,
                    test_interval=None,
                    load_model_paths='',
                    transfer_learning=False,
                    percent_of_data=0.1,
                    verbose=False,
                    number_of_layer=3
                    ):
    if transfer_learning:
        if len(load_model_paths):
            str_latent_dims = load_model_paths.split('latent_dim_')
            latent_dim = int(str_latent_dims[1].split('_')[0])
            print(latent_dim)
            print("Reloading model")
            tmp_model_name = "best_" + "Autoencoder"
            model = Autoencoder(latent_dim=latent_dim).to(device)
            model.load_state_dict(torch.load(os.path.join(load_model_paths, tmp_model_name + ".pth"), device))
            encoder = model.encoder
            classifier_model = Classifier(model_encoder=encoder, latent_dim=latent_dim, transfer_learning=True,
                                          number_of_layer=number_of_layer).to(device)
        else:
            print("Need to provide location for Loading module")

    else:
        classifier_model = Classifier(model_encoder=None, latent_dim=latent_dim, transfer_learning=False,
                                      number_of_layer=number_of_layer).to(device)

    model_name = classifier_model.name()
    print("done loading model", )
    print(classifier_model)
    if len(output_dir):
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H%M")
        final_name = '_'.join((model_name, f"latent_dim_{latent_dim}", f"lr_{learning_rate}",
                               f"BatchSize_{batch_size}", f"Epochs_{num_epochs}", f"data_{percent_of_data * 100}%",
                               time_stamp))
        model_path = os.path.join(output_dir, final_name)
        os.makedirs(model_path, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    all_train_dataset = train_dataset  # list(train_dataset)
    new_train_data_index = []
    prng = np.random.RandomState(42)

    # Balanced sampling of data from each class
    for i in range(10):
        cur_class_index = np.where(np.array(all_train_dataset.targets) == i)[0]
        random_permute = prng.permutation(cur_class_index)[:int(len(cur_class_index) * percent_of_data)]
        new_train_data_index.extend(random_permute)

    train_dataset = torch.utils.data.Subset(train_dataset, new_train_data_index)
    print("size of data set is:", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_interval = int(len(train_loader) / 5) if test_interval is None and int(
        len(train_loader) / 5) > 2 else test_interval
    test_interval = test_interval if int(len(train_loader) / 5) > 2 else 100
    # torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()  # mean square error loss
    optimizer = torch.optim.Adam(classifier_model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)

    outputs = []
    best_test_loss = np.inf
    test_loss = np.inf
    best_f1 = 0
    train_loss_list = []
    test_loss_list = []
    itrrations = []
    best_classifier_model = None
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        itr = 0
        for data in train_loader:
            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                data = next(iter(test_loader))  # get a test batch
            else:
                test_iter = False
            img, lables = data
            img = img.to(device)
            lables = lables.to(device)
            output = classifier_model(img)
            loss = criterion(output, lables)
            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = loss
            else:
                test_loss = loss
            if test_iter and test_loss < 10.0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{int(len(train_dataset) / float(train_loader.batch_size)) + 1}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}"
                )
            test_loss = np.inf
        # end of epoch loss on train and test srt
        epoch_test_loss = []
        for data in test_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            output = classifier_model(img)
            cur_test_loss = criterion(output, labels)
            epoch_test_loss.append(cur_test_loss.cpu().detach().numpy())
        average_test_loss = np.mean(epoch_test_loss)
        epoch_train_loss = []
        for data in train_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            output = classifier_model(img)
            cur_train_loss = criterion(output, labels)
            epoch_train_loss.append(cur_train_loss.cpu().detach().numpy())
        average_train_loss = np.mean(epoch_train_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],"
            f"Train Loss: {float(average_train_loss):.4f}, "
            f"Test Loss: {float(average_test_loss):.4f}"
        )
        if epoch % 3 != 0 or not verbose:
            accuracy, recall, precision, f1, _, _, report = evaluate_model(classifier_model, test_loader, device,
                                                                           verbose=False)
        elif verbose:
            accuracy, recall, precision, f1, figure, figure_normalize, report = evaluate_model(classifier_model,
                                                                                               test_loader, device,
                                                                                               verbose=True)

        itrrations.append(epoch + 1)
        train_loss_list.append(average_train_loss)
        test_loss_list.append(average_test_loss)
        if best_test_loss > average_test_loss:
            best_test_loss = average_test_loss
            print("loss improved: ", best_test_loss)
            best_f1 = f1 if f1 > best_f1 else best_f1
            best_accuracy = accuracy if accuracy > best_accuracy else best_accuracy
            if best_classifier_model is not None:
                del best_classifier_model
            best_classifier_model = copy.deepcopy(classifier_model)
            if len(output_dir):
                torch.save(classifier_model.state_dict(),
                           os.path.join(model_path, "best_" + classifier_model.name() + ".pth"))
        outputs.append((epoch, float(np.mean(epoch_test_loss)), f1, latent_dim), )
        imgs = img.cpu().detach().numpy()[:10]
        recon = output.cpu().detach().numpy()
        reconLabel = recon.argmax(axis=-1)[:10]

        if epoch % 3 == 0 and verbose:
            figure.savefig(
                os.path.join(model_path,
                             classifier_model.name() + f"_epoch_{epoch}_Confusion_matrix_without_normalization" + ".png"))
            figure_normalize.savefig(
                os.path.join(model_path,
                             classifier_model.name() + f"_epoch_{epoch}_Confusion_matrix_with_normalization_epoch_{epoch}" + ".png"))
            cur_plot = plot_imges_vs_prediction(imgs, titles=reconLabel)
            if len(output_dir):
                cur_plot.savefig(
                    os.path.join(model_path, classifier_model.name() + f"_epoch_{epoch}_ImagesVsPredicted" + ".png"))
            else:
                plt.show()
            plt.close(cur_plot)
            plt.close(figure)
            plt.close(figure_normalize)

    accuracy, recall, precision, f1, figure, figure_normalize, report = evaluate_model(best_classifier_model,
                                                                                       train_loader,
                                                                                       device,
                                                                                       verbose=True)
    figure.savefig(
        os.path.join(model_path,
                     classifier_model.name() + "_Confusion_matrix_without_normalization_Train_set" + ".png"))
    figure_normalize.savefig(
        os.path.join(model_path, classifier_model.name() + "_Confusion_matrix_with_normalization_Train_set" + ".png"))
    plt.close(figure_normalize)
    plt.close(figure)

    accuracy, recall, precision, f1, figure, figure_normalize, report = evaluate_model(best_classifier_model,
                                                                                       test_loader,
                                                                                       device,
                                                                                       verbose=True)
    figure.savefig(
        os.path.join(model_path, classifier_model.name() + "_Confusion_matrix_without_normalization" + ".png"))
    figure_normalize.savefig(
        os.path.join(model_path, classifier_model.name() + "_Confusion_matrix_with_normalization" + ".png"))
    plt.close(figure_normalize)
    plt.close(figure)
    with open(os.path.join(model_path, "Model_report" + ".txt"), "w") as text_file:
        print("{}".format(report), file=text_file)

    figure = plot_losses(train_loss_list, test_loss_list, itrrations, 'TrainAndTest_Loss', use_ticks=True)
    if len(output_dir):
        figure.savefig(os.path.join(model_path, classifier_model.name() + "_TrainAndTest_Loss_Plot" + ".png"))
    else:
        plt.show()
    plt.close(figure)

    # open the file in the write mode
    with open(os.path.join(model_path, 'report_csv'), 'w', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(['best_test_loss', 'best_f1', 'best_accuracy'])
        writer.writerow([best_test_loss, best_f1, best_accuracy])
    del train_dataset
    del test_dataset
    del test_loader
    del train_loader
    del classifier_model
    return outputs, best_test_loss, best_f1, best_accuracy


def Q3_train_and_compare_classifiers(percent_of_data=[], output_dir='', num_epochs=25,
                                     batch_size=64,
                                     learning_rate=0.001, train_models_path='', number_of_layer=[3, 4, 6]):
    list_of_dirs = [cur_dir for cur_dir in os.listdir(train_models_path) if
                    os.path.isdir(os.path.join(train_models_path, cur_dir))] if os.path.isdir(
        train_models_path) else None
    full_paths = [os.path.join(train_models_path, cur_model) for cur_model in
                  sorted(list_of_dirs, key=lambda dir_name: int(dir_name.split('latent_dim_')[1].split('_')[0])) if
                  os.path.isdir(os.path.join(train_models_path, cur_model))] if os.path.isdir(
        train_models_path) else train_models_path
    for n in number_of_layer:
        cur_output_dir = os.path.join(output_dir, f"learning_with_{n}_layers")
        os.makedirs(cur_output_dir, exist_ok=True)
        for percent in percent_of_data:
            percent_data_path = os.path.join(cur_output_dir, f"learning_with_{percent * 100}% data")
            os.makedirs(percent_data_path, exist_ok=True)
            transfer_learning_F1 = []
            regular_F1 = []
            transfer_learning_Loss = []
            regular_Loss = []
            latent_spaces_tested = []
            transfer_learning_accuracy = []
            regular_learning_accuracy = []
            for path in full_paths:
                str_latent_dims = path.split('latent_dim_')
                latent_dim = int(str_latent_dims[1].split('_')[0])
                print(latent_dim)
                outputs, best_test_loss, best_f1, best_accuracy = TrainClassifier(latent_dim=12,
                                                                                  output_dir=percent_data_path,
                                                                                  num_epochs=num_epochs,
                                                                                  batch_size=batch_size,
                                                                                  learning_rate=learning_rate,
                                                                                  load_model_paths=path,
                                                                                  transfer_learning=True,
                                                                                  percent_of_data=percent,
                                                                                  number_of_layer=n
                                                                                  )
                del outputs
                transfer_learning_F1.append(best_f1)
                transfer_learning_Loss.append(best_test_loss)
                transfer_learning_accuracy.append(best_accuracy)
                print("finished transfer learning, now on to reg train")
                outputs, best_test_loss, best_f1, best_accuracy = TrainClassifier(latent_dim=latent_dim,
                                                                                  last_filter=64,
                                                                                  output_dir=percent_data_path,
                                                                                  num_epochs=num_epochs,
                                                                                  batch_size=batch_size,
                                                                                  learning_rate=learning_rate,
                                                                                  load_model_paths=path,
                                                                                  transfer_learning=False,
                                                                                  percent_of_data=percent,
                                                                                  number_of_layer=n
                                                                                  )
                del outputs
                regular_F1.append(best_f1)
                regular_Loss.append(best_test_loss)
                regular_learning_accuracy.append(best_accuracy)
                latent_spaces_tested.append(latent_dim)
                print("finished reg learning, Next latent dim\n")
            figure0 = plot_losses(regular_F1, transfer_learning_F1, latent_spaces_tested, 'Test set F1 Vs dim',
                                  label_1="regular_F1", label_2="transfer_learning_F1",
                                  xlabel="latent dim", ylabel="F1")
            figure1 = plot_losses(regular_Loss, transfer_learning_Loss, latent_spaces_tested, 'Test set Loss Vs dim',
                                  label_1="regular_Loss", label_2="transfer_learning_Loss",
                                  xlabel="latent dim", ylabel="Loss")
            figure2 = plot_losses(regular_learning_accuracy, transfer_learning_accuracy, latent_spaces_tested,
                                  'Test set best accuracy Vs dim',
                                  label_1="regular_accuracy", label_2="transfer_learning_accuracy",
                                  xlabel="latent dim", ylabel="accuracy")
            if len(output_dir):
                figure1.savefig(
                    os.path.join(percent_data_path, f"Test set Loss Vs dim percent of data{percent * 100}%" + ".png"))
                figure2.savefig(os.path.join(percent_data_path,
                                             f"Test set accuracy Vs dim percent of data{percent * 100}%" + ".png"))
                figure0.savefig(
                    os.path.join(percent_data_path, f"Test set F1 Vs dim percent of data{percent * 100}%" + ".png"))
            plt.close(figure1)
            plt.close(figure0)
            plt.close(figure2)


def plot_losses(train_losses, test_losses, batch_counts, train_name, label_1="train",
                label_2="test", xlabel="epochs count", ylabel="loss per item", use_ticks=False):
    figure = plt.figure()
    plt.plot(batch_counts, train_losses, 'o--', label=label_1, alpha=0.8)
    plt.plot(batch_counts, test_losses, 'o--', label=label_2, alpha=0.8)
    if use_ticks:
        plt.xticks(range(len(batch_counts)), batch_counts)
    plt.title(f"train and test losses - {train_name}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    return figure


def plot_dim_vs_placeholder(train_losses, test_losses, train_name, label="test"):
    figure = plt.figure()
    plt.plot(train_losses, test_losses, 'o--', label=label, alpha=0.8)

    plt.title(f"test losses - {train_name}")
    plt.xlabel("latent dim")
    plt.ylabel("loss per item")
    plt.legend()
    return figure


def plot_imges_vs_prediction(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i][0], interpolation=None if interp else 'none')
    return f


if __name__ == '__main__':
    pass
    # output_dir = "/home/eldan/Uni/EAModels"
    # latent_dims = [2,4,8,10]
    # Q1_test_latent_dim_result(output_dir = output_dir,  num_epochs = 20,
    #                                                              reload_model=False,
    #                                                              latent_dims=latent_dims,
    #                                                              load_model_paths=output_dir,
    #                                                              number_of_layer=1)

    # output_dir_size = "/home/eldan/Uni/compare_EAModels"
    # output_dir_size = "/home/eldan/Uni/compare_EAModels_test"
    # layer_count = [1,2,4] #
    # latent_dims = [2, 4, 6, 8, 10] #
    # test_different_autoencoders_Q1(latent_dims=latent_dims, output_dir=output_dir_size, layer_count=layer_count, num_epochs=21)

    # Q2_test_Decorrelation_result(output_dir =output_dir, load_model_paths = output_dir, Train_set=False)

    # models_path = output_dir
    # output_dir ="/home/eldan/Uni/classifier_models_second_run"
    # number_of_layer = [1,3,5]
    # Q3_train_and_compare_classifiers(percent_of_data =[0.001,0.005], output_dir=output_dir, num_epochs=30,
    #                 batch_size=128,
    #                 learning_rate=0.001, train_models_path = models_path,number_of_layer=number_of_layer)
