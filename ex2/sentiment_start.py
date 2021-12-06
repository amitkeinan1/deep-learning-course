import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld

# batch_size = 32
# output_size = 2
# hidden_size = 64  # to experiment with
#
# run_recurrent = False  # else run Token-wise MLP
# use_RNN = False  # otherwise GRU
# atten_size = 3  # atten > 0 means using restricted self atten
#
# reload_model = False
# num_epochs = 10
# learning_rate = 0.001
# test_interval = 100

# Loading dataset, use toy = True for obtaining a smaller dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


# Implements RNN Unit
class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid
        self.relu = torch.nn.ReLU()
        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size).to(DEVICE)
        cur_output_size = hidden_size if hidden_size % 2 == 0 else hidden_size + 1
        self.process_output1 = nn.Linear(hidden_size, hidden_size)
        self.layers_list = torch.nn.ModuleList()
        while cur_output_size > 32:
            input_size = cur_output_size
            cur_output_size = int(cur_output_size / 2)
            self.cur_layer = MatMul(input_size, int(cur_output_size))
            self.layers_list.append(self.cur_layer)
        self.in2output = nn.Linear(cur_output_size, output_size).to(DEVICE)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # Implementation of RNN cell
        combined = torch.cat((x.to(DEVICE), hidden_state.to(DEVICE)), 1).to(DEVICE)
        hidden = self.sigmoid(self.in2hidden(combined))
        process_1 = self.relu(self.process_output1(hidden))
        for cur_layer in self.layers_list:
            process_1 = cur_layer(process_1)
        output = self.sigmoid(self.in2output(process_1))[0]

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


# Implements GRU Unit
class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = input_size
        # GRU Cell weights
        self.combined2z = nn.Linear(input_size + hidden_size, hidden_size).to(DEVICE)
        self.combined2r = nn.Linear(input_size + hidden_size, hidden_size).to(DEVICE)
        self.rAndH2hHat = nn.Linear(input_size + hidden_size, hidden_size).to(DEVICE)
        self.hHat2hidden = nn.Linear(input_size + hidden_size, hidden_size).to(DEVICE)
        self.process_output1 = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.process_output2 = nn.Linear(hidden_size, int(hidden_size / 2)).to(DEVICE)
        self.in2output = nn.Linear(int(hidden_size / 2), output_size).to(DEVICE)

        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.relu = torch.nn.ReLU()
        # self.Hhat2H =
        # etc ...

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        x = x.to(DEVICE)
        hidden_state = hidden_state.to(DEVICE)
        combined = torch.cat((x, hidden_state), 1)

        # Implementation of GRU cell
        cur_z = self.sigmoid(self.combined2z(combined))

        cur_r = self.sigmoid(self.combined2r(combined))

        combinedHhat = torch.cat((cur_r * hidden_state, x), 1)
        cur_hHat = self.tanh(self.rAndH2hHat(combinedHhat))

        hidden = (1 - cur_z) * hidden_state + cur_z * cur_hHat

        process1_output = self.relu(self.process_output1(hidden))
        process2_output = self.relu(self.process_output2(process1_output))
        output = self.sigmoid(self.in2output(process2_output))
        # missing implementation

        return output, hidden

    def init_hidden(self, bs):  # if there is a bs
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()
        # # Token-wise MLP network weights

        self.ReLU = torch.nn.ReLU()
        self.sigmoid = torch.sigmoid
        # Token-wise MLP network weights

        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        cur_output_size = hidden_size if hidden_size % 2 == 0 else hidden_size + 1

        self.layers_list = torch.nn.ModuleList()
        while cur_output_size > 32:
            input_size = cur_output_size
            cur_output_size = int(cur_output_size / 2)
            self.cur_layer = MatMul(input_size, int(cur_output_size))
            self.layers_list.append(self.cur_layer)

        self.layer3 = MatMul(cur_output_size, output_size)
        # additional layer(s)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        sub_scores = []
        x = x.to(DEVICE)

        # In case you do not need to transfer the input in layers before going word for word the following
        #  layers should be put in the comment
        # process_x = x
        process_x = self.layer1(x)
        process_x = self.ReLU(process_x)
        process_x = self.layer2(process_x)
        process_x = self.ReLU(process_x)
        for i in range(len(process_x[0])):
            process_x_per_word = process_x[:, i, :]
            for cur_layer in self.layers_list:
                process_x_per_word = cur_layer(process_x_per_word)
                process_x_per_word = self.ReLU(process_x_per_word)
            process_x_per_word = self.layer3(process_x_per_word)
            process_x_per_word = self.sigmoid(process_x_per_word)
            # process_x_per_word = self.ReLU(process_x_per_word)
            sub_scores.append(process_x_per_word)
            # sub_scores.append(process_x)
        final_output = torch.stack(sub_scores).permute((1, 2, 0, 3))
        # final_output = torch.stack(sub_scores)

        # Todo -In case and need to make the sum then transfer it in sigmoid should use the two lines below
        # final_output = final_output[0].sum(axis=1)
        # x = self.sigmoid(final_output)

        x = self.sigmoid(final_output[0])
        # Token-wise MLP network implementation
        # rest

        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)
        self.sigmoid = torch.sigmoid
        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.layers_list = torch.nn.ModuleList()
        # self.cur_layer = MatMul(input_size, int(cur_output_size)).to(DEVICE)
        cur_output_size = hidden_size if hidden_size % 2 == 0 else hidden_size + 1
        while cur_output_size > 32:
            input_size = cur_output_size
            cur_output_size = int(cur_output_size / 2)
            self.cur_layer = MatMul(input_size, int(cur_output_size))
            self.layers_list.append(self.cur_layer)
            self.layers_list.append(self.ReLU)

        self.layer3 = MatMul(cur_output_size, output_size)
        # rest ...

    def name(self):
        return "MLP_atten"

    def forward(self, x):

        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))
        x_nei = []

        vals = self.W_v(x)
        padded_v = pad(vals, (0, 0, atten_size, atten_size, 0, 0))
        v_nei = []

        for k in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))
            v_nei.append(torch.roll(padded_v, k, 1))

        x_nei = torch.stack(x_nei, 2)
        v_nei = torch.stack(v_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]
        vals = v_nei[:, atten_size:-atten_size, :]

        # vals = self.W_v(v_nei)
        keys = self.W_k(x_nei)
        query = self.W_q(x)
        query = torch.stack([query], 3)
        dot_product = torch.matmul(keys, query) / self.sqrt_hidden_size
        dot_product = torch.squeeze(dot_product, dim=-1)
        atten_weights = self.softmax(dot_product)
        atten_weights = torch.stack([atten_weights], 3)
        val_out = atten_weights * vals
        x = torch.sum(val_out, dim=2)

        for cur_layer in self.layers_list:
            x = cur_layer(x)
        x = self.layer3(x)
        x = self.sigmoid(x)

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer
        atten_weights = torch.squeeze(atten_weights)
        return x, atten_weights
