import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
import torch.nn.init as init
import math
import numpy as np


class MotionPredictionNN(nn.Module):
    def __init__(self, n_input, n_output, n_expert_weights, h, drop_prob=0.7):
        super(MotionPredictionNN, self).__init__()
        self.n_expert_weights = n_expert_weights
        # self.fc1 = [nn.Linear(n_input, h)]*n_expert_weights
        # self.fc2 = [nn.Linear(h,h)]*n_expert_weights
        # self.fc3 = [nn.Linear(h, n_output)]*n_expert_weights
        self.n_input = n_input
        self.n_output = n_output
        self.h = h
        self.expert_weights_fc0 = Parameter(torch.Tensor(n_expert_weights, h, n_input))
        self.expert_weights_fc1 = Parameter(torch.Tensor(n_expert_weights, h, h))
        self.expert_weights_fc2 = Parameter(torch.Tensor(n_expert_weights, n_output, h))
        self.expert_bias_fc0 = Parameter(torch.Tensor(n_expert_weights, h))
        self.expert_bias_fc1 = Parameter(torch.Tensor(n_expert_weights, h))
        self.expert_bias_fc2 = Parameter(torch.Tensor(n_expert_weights, n_output))
        self.reset_parameters()

        self.drop_prob = drop_prob

    def reset_parameters(self):
        init.kaiming_uniform_(self.expert_weights_fc0, a=math.sqrt(5))
        init.kaiming_uniform_(self.expert_weights_fc1, a=math.sqrt(5))
        init.kaiming_uniform_(self.expert_weights_fc2, a=math.sqrt(5))
        init.zeros_(self.expert_bias_fc0)
        init.zeros_(self.expert_bias_fc1)
        init.zeros_(self.expert_bias_fc2)

    def forward(self, x, BC):
        W0, B0, W1, B1, W2, B2 = self.blend(BC)

        x = F.dropout(x, self.drop_prob, self.training)
        x = torch.baddbmm(B0.unsqueeze(2), W0, x.unsqueeze(2))
        x = F.elu(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = torch.baddbmm(B1.unsqueeze(2), W1, x)
        x = F.elu(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = torch.baddbmm(B2.unsqueeze(2), W2, x)
        x = x.squeeze(2)
        return x

    def blend(self, BC):
        BC_w = BC.unsqueeze(2).unsqueeze(2)
        BC_b = BC.unsqueeze(2)

        W0 = torch.sum(BC_w * self.expert_weights_fc0.unsqueeze(0), dim=1)
        B0 = torch.sum(BC_b * self.expert_bias_fc0.unsqueeze(0), dim=1)
        W1 = torch.sum(BC_w * self.expert_weights_fc1.unsqueeze(0), dim=1)
        B1 = torch.sum(BC_b * self.expert_bias_fc1.unsqueeze(0), dim=1)
        W2 = torch.sum(BC_w * self.expert_weights_fc2.unsqueeze(0), dim=1)
        B2 = torch.sum(BC_b * self.expert_bias_fc2.unsqueeze(0), dim=1)
        return W0, B0, W1, B1, W2, B2


class GatingNN(nn.Module):
    def __init__(self, n_input, n_expert_weights, hg, drop_prob=0.0):
        super(GatingNN, self).__init__()
        self.fc0 = nn.Linear(n_input, hg)
        self.fc1 = nn.Linear(hg, hg)
        self.fc2 = nn.Linear(hg, n_expert_weights)

        self.drop_prob = drop_prob

    def forward(self, x):
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.elu(self.fc0(x))
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class MANN(nn.Module):
    def __init__(self, index_gating, n_expert_weights, hg, n_input_motion, n_output_motion, h, drop_prob_gat=0.0, drop_prob_mot=0.3):
        super(MANN, self).__init__()
        self.index_gating = index_gating
        n_input_gating = self.index_gating.shape[0]
        self.gatingNN = GatingNN(n_input_gating, n_expert_weights, hg, drop_prob_gat)
        self.motionNN = MotionPredictionNN(n_input_motion, n_output_motion, n_expert_weights, h, drop_prob_mot)

    def forward(self, x):
        in_gating = x[..., self.index_gating]
        BC = self.gatingNN(in_gating)
        return self.motionNN(x, BC)
