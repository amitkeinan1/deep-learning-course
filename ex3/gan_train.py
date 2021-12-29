import os

from ae_models import Autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_ae():
    latent_dim = 32
    ae_path = os.path.join("ae_models", "AE_model_latent_dim_32_lr_0.001_BatchSize_64_Epochs_20_2021-12-27_1113",
                           "best_Autoencoder.pth")
    model = Autoencoder(latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ae_path), DEVICE))
    return model

def train_gan():
    train_loader, test_loader = get_data()


if __name__ == '__main__':
