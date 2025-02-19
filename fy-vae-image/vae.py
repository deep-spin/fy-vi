from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from losses import entmax_loss, entmax_bisect
import optuna
from contextlib import contextmanager
from pathlib import Path
import multiprocessing
import sys
import os

torch.manual_seed(42)

LOSS=sys.argv[1]

@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


class GpuQueue:
    def __init__(self, N_GPUS=4):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train(params, gpu_id=0, valid_size=5000, eval_interval=50):

    torch.cuda.set_device(gpu_id)
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    
    mnist_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

    # Generate indices
    num_samples = len(mnist_dataset)
    indices = list(range(num_samples))

    # Split indices into training and validation
    train_indices = indices[:-valid_size]  # All but the last `valid_size` indices
    valid_indices = indices[-valid_size:]   # Last `valid_size` indices
    train_dataset = torch.utils.data.Subset(mnist_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(mnist_dataset, valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=params['batch_size'], shuffle=False, **kwargs)
    
    model = VAE().cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    exp_name = f"experiments/lr{params['learning_rate']:.2e}_bs{params['batch_size']}_alpha{params['alpha_ent']}_np{params['n_epochs']}_kl{params['kld_weight']:.2e}_{LOSS}"
    os.makedirs(exp_name, exist_ok=True)

    for epoch in range(1, params['n_epochs'] + 1):

        model.train()
        train_loss = 0
        best_valid_loss = float('inf')

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            if LOSS=="sigmoid":
                loss = F.binary_cross_entropy(torch.sigmoid(recon_batch), data.view(-1, 784), reduction='sum')
            else:
                recon_batch_3d = torch.stack([recon_batch, torch.zeros_like(recon_batch)], axis=-1)
                data_3d = torch.stack([data.view(-1, 784), 1 - data.view(-1, 784)], axis=-1) #torch.zeros_like(data.view(-1, 784))
                loss = entmax_loss(recon_batch_3d, data_3d, params['alpha_ent']).sum()
           
            loss_final = loss + params['kld_weight'] * KLD
            loss_final.backward()
            train_loss += loss_final.item()
            optimizer.step()

            if batch_idx % eval_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data)))
                valid_loss = eval(valid_loader, model, params['alpha_ent'])
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                print('====> Epoch: {} Average Valid loss: {:.4f} Best Valid Loss: {:.4f}'.format(
                    epoch, valid_loss, best_valid_loss))
        
        test_loss = eval(test_loader, model, params['alpha_ent'])
        with torch.no_grad():
            sample = torch.randn(64, 20).cuda()
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                        f'{exp_name}/sample_' + str(epoch) + '.png')
        print('====> Epoch: {} Average Test loss: {:.4f}'.format(epoch, test_loss))
            
    return best_valid_loss

def eval(test_loader, model, alpha):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            data = data.cuda()
            recon_batch, _, _ = model(data)
            if LOSS == "sigmoid":
                test_loss += torch.abs(data.view(-1, 784)-torch.sigmoid(recon_batch)).sum()
            else:
                test_loss += torch.abs(data.view(-1, 784)
                                       -entmax_bisect(torch.stack([recon_batch, torch.zeros_like(recon_batch)], axis=-1), alpha=alpha, dim=-1)[:,:,0]).sum()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss
    

class Objective:

    def __init__(self, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial: optuna.trial.Trial):
    
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'kld_weight': trial.suggest_categorical('kld_weight', [1e-4, 1e-3, 1e-2, 0.1, 1.0]),
            'n_epochs': trial.suggest_categorical('n_epochs', [5, 10, 20]),
            'alpha_ent': trial.suggest_loguniform('alpha_ent', 0.1, 2),
        }
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            best_val_loss = train(params, gpu_id=gpu_i)
            return best_val_loss

if __name__ == "__main__":
    exp_dir = Path("experiments/")
    tuning_log_dir = exp_dir / 'tuning_logs'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    while tuning_logs.exists():
        i += 1
        tuning_logs = tuning_log_dir / f'logs_{i}.txt'

    with log_stdout(tuning_logs):
        study = optuna.create_study(study_name=LOSS, direction="minimize",
                                    storage=f'sqlite:///{tuning_log_dir}/{LOSS}.db', load_if_exists=True)
        study.optimize(Objective(GpuQueue()), n_trials=50, n_jobs=4, show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    