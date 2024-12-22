# For CIFAR10 vae structure
import torch
from torch import nn
from torch.nn import functional as F

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    # print(idx)
    # print(idx.shape)
    # print(type(idx))
    onehot.scatter_(1, idx.type(torch.cuda.LongTensor), 1)
    # print(onehot.shape)
    return onehot

class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_sizes=None):
        super(VAE, self).__init__()
        self.num_classes = input_size

        self.latent_size = latent_size

        modules = []
        if hidden_sizes == None:
            hidden_sizes = [32, 64, 128]
        
        # Build encoder
        layer_input_size = input_size*2
        for hidden_size in hidden_sizes:
            modules.append(
                nn.Sequential(
                    nn.Linear(layer_input_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(True),
                )
            )
            layer_input_size = hidden_size
        
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_size)
        self.fc_var = nn.Linear(hidden_sizes[-1], latent_size)

        self.decoder_input = nn.Linear(latent_size+input_size, hidden_sizes[-1])

        modules = []
        hidden_sizes.reverse()
        # hidden_sizes[0] += input_size

        # Build decoder
        for i in range(len(hidden_sizes)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                    nn.BatchNorm1d(hidden_sizes[i+1]),
                    nn.ReLU(True),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_sizes[-1], input_size)

    def encode(self, input, class_idx):
        # print(input.size())
        input = torch.cat((input, idx2onehot(class_idx, self.num_classes)),dim=1)
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, class_idx):
        z = torch.cat((z, idx2onehot(class_idx, self.num_classes)),dim=1)
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        # return torch.randn_like(mu)

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_size)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def sample_condition(self, num_samples, class_idx, device):
        z = torch.randn(num_samples, self.latent_size)
        z = z.to(device)
        samples = self.decode(z, class_idx)
        return samples

    def forward(self, input, class_idx, release = 'softmax'):
        mu, log_var = self.encode(input, class_idx)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z, class_idx)
        if release=='softmax':
            result = F.softmax(result, dim=1)
        elif release=='log_softmax':
            result = F.log_softmax(result, dim=1)
        elif release=='raw':
            result = result
        else:
            raise Exception("=> Wrong release flag!!!")
        return result, mu, log_var

    def generate(self, class_idx, device, release = 'softmax'):
        if (type(class_idx) is int):
            class_idx = torch.tensor([class_idx])
        class_idx = class_idx.to(device)
        # print(class_idx)
        batch_size = class_idx.shape[0]
        z = torch.randn((batch_size, self.latent_size)).to(device)

        prob = idx2onehot(class_idx, self.num_classes)
        mu, log_var = self.encode(prob, class_idx)
        z = self.reparameterize(mu, log_var)

        result = self.decode(z, class_idx)
        if release=='softmax':
            return F.softmax(result, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(result, dim=1)
        elif release=='raw':
            return result
        else:
            raise Exception("=> Wrong release flag!!!")


def load_vae(vae_model, path):
    try:
        checkpoint = torch.load(path)
        vae_model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print("=> loaded vae_model checkpoint '{}' (epoch {}, loss {:.4f})".format(path, epoch, best_loss))
        return 1
    except Exception as e:
        print(e)
        print("=> load vae_model checkpoint '{}' failed".format(path))
        return 0
    # return vae_model
