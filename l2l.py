import torch
import torch.nn as nn
import torch.nn.functional as F

class L2L(nn.Module):
    '''
        define Learning-2-Learn class with LSTM architecture
    '''
    
    def __init__(self, circ_function, num_feats=3, batch_size=16):
        # circ_function should be a function which is a pennylane qnode

        super().__init__()

        self.num_feats = num_feats  # rnn_output, qnn input params
        self.batch_size = batch_size
        # does pennylane support circuits that return multiple measurements?
        self.rnn_input_size = 1 # qnn output size
        self.function = circ_function

        # the target is required
        self.target = None
        self.hid_cell = None
        self.rnn_output = None
        self.qnn_output = None

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size, hidden_size=self.num_feats, num_layers=1, dropout=0
        )

    def init_hid_cell(self, seq_len=1):
        # concatenate and store all the output tensors here
        self.rnn_output = torch.tensor([])
        self.qnn_output = torch.zeros(seq_len, self.batch_size, self.rnn_input_size)
        
        hidden = torch.zeros(seq_len, self.batch_size, self.num_feats)
        cell = torch.zeros(seq_len, self.batch_size, self.num_feats)
        self.hid_cell = (hidden, cell)

    def step(self):
        assert self.hid_cell is not None
    
        x = self.qnn_output[[-1], :, :]
        # print(f'RNN input {x.shape}')
        
        rnn_output, self.hid_cell = self.lstm(x, self.hid_cell)
        self.rnn_output = torch.cat((self.rnn_output, rnn_output), dim=0)  # dims are : (seq_dim, batch_size, feature_size)
        # print(f'RNN output: {rnn_output.shape} RNN hist {self.rnn_output.shape}')
        
        assert rnn_output.shape[0] == 1
        qnn_output = torch.zeros_like(x)
        # the pennylane qnode can't handle batching; iterate through the batch one at a time
        for i in range(rnn_output.shape[1]):
            qnn_input_batch_element = rnn_output[0, i, :]
            qnn_output_batch_element = self.function(qnn_input_batch_element)
            assert qnn_output_batch_element.nelement() == self.rnn_input_size
            qnn_output[0, i, :] = qnn_output_batch_element
        
        # subtract target value so that loss is simply minimized at 0
        qnn_output[0,:,:] = qnn_output[0,:,:] - self.target
        # print(f'circuit output: {qnn_output.shape}')
        self.qnn_output = torch.cat((self.qnn_output, qnn_output), dim=0)

        return self.qnn_output

    def loss(self, true=None):
        # compare the qnn output to the given target ('true')

        # print(f'true: {true.shape}, pred: {self.qnn_output.shape}')
        
        if true==None:
            true = torch.zeros(self.qnn_output.shape)
            
        assert true.shape == self.qnn_output.shape
        
        return F.mse_loss(self.qnn_output, true)

    def numpy_output(self):
        return self.qnn_output.detach().numpy().squeeze()
