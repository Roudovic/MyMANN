from __future__ import print_function
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


#Global variables
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)



class Trainer:
    def __init__(self, mann, dataset, lr, decay_rate, batch_size, n_epoch, savepath):
        self.mann = mann
        self.lr = lr
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.optimizer = optim.AdamW(self.mann.parameters(), lr, weight_decay=decay_rate)
        # self.optimizer = optim.Adam(self.mann.parameters(), lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, 2)
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
        self.lossF = nn.MSELoss()
        self.savepath = savepath

    def train(self):
        writer = SummaryWriter('runs/exp-1expert')
        cum_loss = 0
        k = 0
        k_tot = 0
        iter = len(self.dataloader)
        for epoch in range(self.n_epoch):
            for data in self.dataloader:
                self.mann.zero_grad()

                pred = self.mann(data["input"].to(device))
                # print(pred)
                loss = self.lossF(pred, data["output"].to(device))
                loss.backward()
                # for param in self.mann.parameters():
                #     print(param.grad.data.sum())
                self.optimizer.step()
                
                cum_loss += loss.item()
                k += 1
                k_tot += 1

                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch * len(self.dataloader) + k)

                if (k_tot % 1000 == 1):
                    self.save_gating_weights()
                    self.save_motion_weights()
                    avg_sum = cum_loss / k
                    k = 0
                    cum_loss = 0
                    print("Epoch ", epoch)
                    print("Step ", k_tot)
                    print("Loss :", avg_sum)
                    print("________________________\n")
                    # log_to_file("loss_log.txt", ( "Loss : " + loss.item()))
            
            self.scheduler.step()
            avg_sum = cum_loss / k
            k = 0
            cum_loss = 0

    def save_gating_weights(self):
        nn_path = self.savepath + '/nn/'
        wc0_w = self.mann.gatingNN.fc0.weight.data.cpu().numpy()
        wc0_w.tofile(nn_path + 'wc0_w.bin')
        wc0_b = self.mann.gatingNN.fc0.bias.data.cpu().numpy()
        wc0_b.tofile(nn_path + 'wc0_b.bin')

        wc1_w = self.mann.gatingNN.fc1.weight.data.cpu().numpy()
        wc1_w.tofile(nn_path + 'wc1_w.bin')
        wc1_b = self.mann.gatingNN.fc1.bias.data.cpu().numpy()
        wc1_b.tofile(nn_path + 'wc1_b.bin')

        wc2_w = self.mann.gatingNN.fc2.weight.data.cpu().numpy()
        wc2_w.tofile(nn_path + 'wc2_w.bin')
        wc2_b = self.mann.gatingNN.fc2.bias.data.cpu().numpy()
        wc2_b.tofile(nn_path + 'wc2_b.bin')

    def save_motion_weights(self):
        nn_path = self.savepath + '/nn/'
        for i in range(self.mann.motionNN.n_expert_weights):
            cp0_a = self.mann.motionNN.expert_weights_fc0[i].data.cpu().numpy()
            cp0_a.tofile(nn_path + 'cp0_a' + str(i) + '.bin')
            cp0_b = self.mann.motionNN.expert_bias_fc0[i].data.cpu().numpy()
            cp0_b.tofile(nn_path + 'cp0_b' + str(i) + '.bin')

            cp1_a = self.mann.motionNN.expert_weights_fc1[i].data.cpu().numpy()
            cp1_a.tofile(nn_path + 'cp1_a' + str(i) + '.bin')
            cp1_b = self.mann.motionNN.expert_bias_fc1[i].data.cpu().numpy()
            cp1_b.tofile(nn_path + 'cp1_b' + str(i) + '.bin')

            cp2_a = self.mann.motionNN.expert_weights_fc2[i].data.cpu().numpy()
            cp2_a.tofile(nn_path + 'cp2_a' + str(i) + '.bin')
            cp2_b = self.mann.motionNN.expert_bias_fc2[i].data.cpu().numpy()
            cp2_b.tofile(nn_path + 'cp2_b' + str(i) + '.bin')


def log_to_file(filename, message):
    with open(filename, 'a') as f:
        f.write(message)

def main():

    motionDataset = MotionCaptureDataset("Data")
    print(len(motionDataset))
    n_input = motionDataset.n_features_input
    n_output = motionDataset.n_features_output
    print(n_input)
    print(n_output)
    
    total_batch = len(motionDataset)/32
    weightDecayNormalized = 0.0025 / pow(total_batch*10, 0.5)

    # index_gating = np.array([10, 15, 19, 23])
    index_gating_noStyle = np.array([165,166,167,
                                     225,226,227,
                                     273,274,275,
                                     321,322,323,
                                     48])
    index_gating = np.array([285, 286, 287,
                             345, 346, 347,
                             393, 394, 395,
                             441, 442, 443,
                             84, 85, 86, 87, 88, 89, 90])

    t0 = time.time()
    mann = MANN(index_gating,
                n_expert_weights=1,
                hg=32,
                n_input_motion=n_input,
                n_output_motion=n_output,
                h=512,
                drop_prob_gat = 0.3,
                drop_prob_mot=0.3).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        mann = nn.DataParallel(mann, list(range(ngpu)))

    trainer = Trainer(mann, motionDataset, 0.0001, weightDecayNormalized, batch_size=32, n_epoch=70, savepath="Data/nn_simple")
    trainer.train()
    print("Training  took ", (time.time()- t0), "seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
