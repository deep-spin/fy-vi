import os

import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from losses import entmax_loss, entmax15
import optuna
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from helper import log_stdout
import sys
import os
from contextlib import contextmanager
import multiprocessing
from spcdist.torch import MultivariateBetaGaussianDiag

LOSS=sys.argv[1]
N_GPUS = 4

class GpuQueue:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


#-------------------------------
def data_set(data_url):
    """process data input."""
    data = []
    word_count = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()
        doc = {}
        count = 0
        for id_freq in id_freqs[1:]:
            items = id_freq.split(':')
            # python starts from 0
            doc[int(items[0])-1] = int(items[1])
            count += int(items[1])
        if count > 0:
            data.append(doc)
            word_count.append(count)
    fin.close()
    return data, word_count


def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in list(range(data_size // batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding
    return batches


def fetch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            count_batch.append(count[doc_id])
            mask[i]=1.0
        else:
            count_batch.append(0)
    return data_batch, count_batch, mask


#-------------------------------
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.mlp = nn.Linear(args['n_input'], args['n_hidden'])
        self.mean_fc = nn.Linear(args['n_hidden'], args['n_topics'])
        self.logsigm_fc = nn.Linear(args['n_hidden'], args['n_topics'])
        nn.init.zeros_(self.logsigm_fc.weight)  # cf. https://github.com/ysmiao/nvdm/blob/master/nvdm.py#L51
        nn.init.zeros_(self.logsigm_fc.bias)

    def forward(self, doc_freq_vecs, mask):

        en_vec = F.tanh(self.mlp(doc_freq_vecs))
        mean = self.mean_fc(en_vec)
        logsigm = self.logsigm_fc(en_vec) 
        
        if self.args['normal']=='normal':
            kld = -0.5 * torch.sum(1 - torch.square(mean) + 2 * logsigm - (2 * logsigm).exp(), 1)
            return mask*kld, mean, logsigm, None

        else:
            sigma2 = torch.exp(logsigm)
            mbg = MultivariateBetaGaussianDiag(mean, sigma2, alpha=self.args['beta'])
            r = torch.exp(mbg.log_radius).cuda()
            n = mbg._fact_scale.rank 

            first_term = 0.5 * torch.sum(mean ** 2, dim=1) 
            numerator = r ** 2
            denominator = 2 * self.args['beta'] + n * (self.args['beta'] - 1)  

            determinant = torch.prod(sigma2, dim=1)  # [64]
            exponent = -1 / (n + 2 / (self.args['beta'] - 1))  

            coefficient = 0.5 * (self.args['beta'] - 1) 

            second_term = (numerator / denominator) * (determinant + 1e-8 )  ** exponent * (
                (1 + coefficient * torch.sum(sigma2, dim=1)) -
                (1 + 0.5 * n * (self.args['beta'] - 1))  )
            
            fy = first_term + second_term  
            return mask*fy, mean, logsigm, mbg      


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.decoder = nn.Linear(args['n_topics'], args['n_input'])  #

    def forward(self, doc_freq_vecs, mean, logsigm, mvbg=None, train=True):
        if train:
            if self.args['normal'] == "normal":      
                eps = torch.randn(self.args['batch_size'], self.args['n_topics']).cuda()
                doc_vec = torch.mul(torch.exp(logsigm), eps) + mean
            else:
                assert mvbg is not None
                doc_vec = mvbg.rsample(sample_shape=(1,))[0] 
            
            z = self.decoder(doc_vec)
            
            if self.args['loss'] == 'entmax':
                recon = entmax_loss(z, torch.tensor((doc_freq_vecs.T/doc_freq_vecs.sum(-1)), 
                                                              dtype=torch.float).T, alpha=self.args['alpha'])*doc_freq_vecs.sum(-1) # 1.01 for sanity check
            else:
                logprobs = F.log_softmax(z, dim=1)
                recon = -torch.sum(torch.mul(logprobs, doc_freq_vecs), 1)
            return recon
        else:
            z = self.decoder(mean)
            temp = torch.tensor((doc_freq_vecs.T/doc_freq_vecs.sum(-1)), 
                                                              dtype=torch.float).T
            if self.args['loss'] == 'entmax':
                recon =  torch.abs(temp-entmax15(z, dim=-1)).sum() 
            else:
                recon =  torch.abs(temp-torch.softmax(z, dim=-1)).sum() 
            return recon


#-------------------------------
def make_optimizer(encoder, decoder, args):
    if args['optimizer'] == 'Adam':
        optimizer_enc = torch.optim.Adam(encoder.parameters(), args['learning_rate'], betas=(args['momentum'], 0.999))
        optimizer_dec = torch.optim.Adam(decoder.parameters(), args['learning_rate'], betas=(args['momentum'], 0.999))

    elif args['optimizer'] == 'SGD':
        optimizer_enc = torch.optim.SGD(encoder.parameters(), args['learning_rate'], momentum=args['momentum'])
        optimizer_dec = torch.optim.SGD(decoder.parameters(), args['learning_rate'], momentum=args['momentum'])

    return optimizer_enc, optimizer_dec

def run_eval(data_set, count, batches, encoder, decoder):
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        error_sum = 0.0
        # ppx_sum = 0.0
        kld_sum = 0.0
        word_count = 0
        doc_count = 0

        for idx_batch in batches:
            data_batch, count_batch, mask = fetch_data(data_set, count, idx_batch, 2000)
            data_batch = torch.tensor(data_batch, dtype=torch.float).cuda()
            count_batch = torch.tensor(count_batch, dtype=torch.float).cuda()
            mask = torch.tensor(mask).cuda()

            kld, mean, logsigm, mbg = encoder(data_batch, mask)
            objective = decoder(data_batch, mean, logsigm, mbg, False)

            # loss = objective + kld 

            error_sum += torch.sum(objective)
            kld_sum += (kld.sum() / mask.sum())

            word_count += torch.sum(count_batch)
            count_batch = torch.add(count_batch, 1e-12)
            # ppx_sum += (loss/count_batch).sum()
            doc_count += mask.sum()

        # print_kld = kld_sum / len(batches)
        # print_vppx = torch.exp(loss_sum / word_count)
        # print_ppx_perdoc = torch.exp(ppx_sum / doc_count)

    return error_sum / len(batches), kld_sum / len(batches)

#-------------------------------
def train(params, data_dir, gpu_id=0, log_every=100):

    torch.cuda.set_device(gpu_id)

    args = {'batch_size': params['batch_size'],
              'optimizer': 'Adam',
              'learning_rate': params['learning_rate'],
              'momentum': 0.9,
              'n_epoch': params['n_epoch'],
              'n_alternating_epoch': params['n_alternating_epoch'],
              'init_mult': 0, # 0.001,
              'n_input': 2000,
              'n_topics': params['n_topics'], 
              'n_hidden': 500,
              'kld_weight': params['kld_weight'],
              'seed': 42,
              'loss': LOSS,
              'alpha': params['alpha_ent'],
              'normal': params['normal'],
              'beta': params['beta'], 
              }

    # exp_name = f"experiments/lr{params['learning_rate']:.2e}_bs{params['batch_size']}_alpha{params['alpha_ent']}_ae{params['n_alternating_epoch']}_np{params['n_epoch']}_kl{params['kld_weight']:.2e}_{LOSS}"
    # os.makedirs(exp_name, exist_ok=True)
    # log_file = open(f"{exp_name}/logs.txt", "w")

    torch.manual_seed(args['seed'])

    train_url = os.path.join(data_dir, 'train.feat')
    train_set, train_count = data_set(train_url)

    # test, dev batches
    test_url = os.path.join(data_dir, 'test.feat')
    test_set, test_count = data_set(test_url)
    dev_set = test_set[:1000]
    dev_count = test_count[:1000]
    test_set = test_set[1000:]
    test_count = test_count[1000:]
    
    test_batches = create_batches(len(test_set), len(test_set), shuffle=False)
    dev_batches = create_batches(len(dev_set), len(dev_set), shuffle=False)

    # model
    encoder = Encoder(args)
    encoder.cuda()
    decoder = Decoder(args)
    decoder.cuda()

    optimizer_enc, optimizer_dec = make_optimizer(encoder, decoder, args)
    best_error = 10000
    #-------------------------------
    # train
    for epoch in tqdm(range(args['n_epoch'])):
        train_batches = create_batches(len(train_set), args['batch_size'], shuffle=True)
        for switch in list(range(0, 2)):
            if switch == 0:
                optimizer = optimizer_dec
                decoder.train()
                print_mode = 'updating decoder'
            else:
                optimizer = optimizer_enc
                encoder.train()
                print_mode = 'updating encoder'
            for i in list(range(args['n_alternating_epoch'])):
                ppx_sum = 0.0
                kld_sum = 0.0
                word_count = 0
                doc_count = 0
                loss_sum = 0.0

                for idx_batch in train_batches[:-1]:
                    data_batch, count_batch, mask = fetch_data(train_set, train_count, idx_batch, 2000)
                    data_batch = torch.tensor(data_batch, dtype=torch.float).cuda()
                    count_batch = torch.tensor(count_batch, dtype=torch.float).cuda()
                    mask = torch.tensor(mask).cuda()

                    kld, mean, logsigm, mvbg = encoder(data_batch, mask)
                    objective = decoder(data_batch, mean, logsigm, mvbg)

                    loss = objective + args['kld_weight'] * kld

                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    loss_sum += torch.sum(loss)
                    kld_sum += (kld.sum() / mask.sum())
                    
                    word_count += torch.sum(count_batch)
                    # per document loss
                    count_batch = torch.add(count_batch, 1e-12)
                    ppx_sum += (loss.detach()/count_batch).sum()
                    doc_count += mask.sum()

                print_kld = kld_sum/len(train_batches[:-1])
                print_ppx = torch.exp(loss_sum / word_count)
                print_ppx_perdoc = torch.exp(ppx_sum / doc_count)

                if epoch % log_every==0:
                    print(f'| Epoch train: {epoch+1} | {i} | {print_mode} | Corpus ppx: {print_ppx:.5f} | Per doc ppx: {print_ppx_perdoc:.5f} | KLD: {print_kld:.5f}\n')
        
        #-------------------------------
        # dev
        if epoch % log_every==0:
            error, print_kld = run_eval(dev_set, dev_count, dev_batches, encoder, decoder)
            print(f'| Epoch eval: {epoch+1}| Reconst error: {error:.5f} | KLD: {print_kld:.5}\n')
            if error < best_error:
                best_error = error
                # torch.save(encoder.state_dict(), f"{exp_name}/encoder_{epoch}.pkl")
                # torch.save(decoder.state_dict(), f"{exp_name}/decoder_{epoch}.pkl")
                # with open(f"{exp_name}/weights_{epoch}.pkl", "wb") as f:
                #     pickle.dump(decoder.decoder.weight.cpu().data, f)
            
    #-------------------------------
    # test
    test_error, print_kld = run_eval(test_set, test_count, test_batches, encoder, decoder)
    print(f'| Epoch test: {epoch+1} | Reconst error: {test_error:.5f}| KLD: {print_kld:.5}\n')
    # log_file.close()
    return error


class Objective:

    def __init__(self, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial: optuna.trial.Trial):
        data_dir='./nvdm/data/20news/'
    
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-6, 5e-6, 1e-5, 5e-5]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 256, 1024, 4096]),
            'kld_weight': trial.suggest_categorical('kld_weight', [1e-4, 1e-3, 1e-2, 0.1, 1.0]),
            'n_epoch': trial.suggest_categorical('n_epoch', [500, 1000]),
            'n_alternating_epoch': trial.suggest_categorical('n_alternating_epoch', [1, 3, 5, 7, 9]),
            'alpha_ent': trial.suggest_loguniform('alpha_ent', 1, 4),
            'n_topics': trial.suggest_categorical('n_topics', [20, 50]),
            'beta':  trial.suggest_loguniform('alpha_ent', 1, 3),
        }
        params['normal'] = 'q-normal'
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            best_val_loss = train(params, data_dir, gpu_id=gpu_i)
            return best_val_loss

def main():
    # data_dir='./nvdm/data/20news/'
    # params = {
    #     'learning_rate': 1e-4, #5e-5
    #     'batch_size': 1024,
    #     'kld_weight': 0.03,
    #     'n_epoch': 1000,
    #     'alpha_ent': 1.5, 
    #     'n_alternating_epoch': 3,
    #     'n_topics': 50,
    #     'normal': 'q-normal',
    #     'beta': 2,
    # }
    # train(params, data_dir)
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
                                    storage=f'sqlite:///{tuning_log_dir}/q{LOSS}.db', load_if_exists=True)
        study.optimize(Objective(GpuQueue()), n_trials=50, n_jobs=4, show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()


