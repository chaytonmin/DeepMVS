
import torch.nn as nn
import torch
import numpy as np

from .convlstm import *
from .submodule import *
#from module import *

class FeatNet(nn.Module):
    def __init__(self, gn):
        super(FeatNet, self).__init__()
        base_filter = 8
        if gn:
            self.conv0_0 = convgnrelu(3, base_filter * 2, kernel_size=3, stride=1, dilation=1)
            self.conv0_1 = convgnrelu(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
            self.conv0_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=2)
            self.conv0_3 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv1_2 with conv0_2
            self.conv1_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=3)
            self.conv1_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv2_2 with conv0_2
            self.conv2_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=4)
            self.conv2_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # with concat conv0_3, conv1_2, conv2_2
            self.conv = nn.Conv2d(base_filter * 12, base_filter*4, 3, 1, 1)
        else:
            # in_channels, out_channels, kernel_size=3, stride=1, pad=1
            # in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
            self.conv0_0 = convbn(3, base_filter * 2, kernel_size=3, stride=1, dilation=1)
            self.conv0_1 = convbn(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
            self.conv0_2 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=2)
            self.conv0_3 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv1_2 with conv0_2
            self.conv1_1 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=3)
            self.conv1_2 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv2_2 with conv0_2
            self.conv2_1 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=4)
            self.conv2_2 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            self.conv = nn.Conv2d(base_filter * 12, base_filter*4, 3, 1, 1)

    def forward(self, x):
        
        conv0_0 = self.conv0_0(x)
        conv0_1 = self.conv0_1(conv0_0)
        conv0_2 = self.conv0_2(conv0_1)
        conv0_3 = self.conv0_3(conv0_2)

        conv1_2 = self.conv1_2(self.conv1_1(conv0_2))

        conv2_2 = self.conv2_2(self.conv2_1(conv0_2))

        conv = self.conv(torch.cat([conv0_3, conv1_2, conv2_2], 1))
        return conv

class FeatNet_new(nn.Module):
    def __init__(self, gn):
        super(FeatNet, self).__init__()
        base_filter = 8
        if gn:
            self.conv0_0 = convgnrelu(3, base_filter, kernel_size=3, stride=1, dilation=1)
            self.conv0_1 = convgnrelu(base_filter, base_filter * 2, kernel_size=3, stride=1, dilation=1)
            self.conv0_2 = convgnrelu(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
            
            self.conv1_0 = convgnrelu(base_filter * 4, base_filter * 8, kernel_size=3, stride=1, dilation=2)
            self.conv1_1 = convgnrelu(base_filter * 8, base_filter * 8, kernel_size=3, stride=1, dilation=1)
            self.conv1_2 = convgnrelu(base_filter * 8, base_filter * 8, kernel_size=3, stride=1, dilation=1)
            
            # with concat conv0_3, conv1_2, conv2_2
            self.conv = nn.Conv2d(base_filter * 8, base_filter*4, 3, 1, 1)
   

    def forward(self, x):
        
        x = self.conv0_2(self.conv0_1(self.conv0_0(x)))
        x = self.conv1_2(self.conv1_1(self.conv1_0(x)))
        
        conv = self.conv(x)
        return conv

class FeatNet2(nn.Module):
    def __init__(self, gn):
        super(FeatNet, self).__init__()
        base_filter = 8
        if gn:
            self.conv_0 = convgnrelu(3, base_filter, kernel_size=3, stride=1, dilation=1)
            self.conv_1 = convgnrelu(base_filter , base_filter * 2, kernel_size=3, stride=1, dilation=1)
            self.conv_2 = convgnrelu(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=2)
        
            self.conv0_0 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=5, stride=2, dilation=1)
            self.conv0_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)
            self.conv0_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=2)
            self.conv0_3 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv1_2 with conv0_2
            self.conv1_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=3)
            self.conv1_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv2_2 with conv0_2
            self.conv2_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=4)
            self.conv2_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # with concat conv0_3, conv1_2, conv2_2
            self.conv = nn.Conv2d(base_filter * 12, base_filter*4, 3, 1, 1)
        else:
            # in_channels, out_channels, kernel_size=3, stride=1, pad=1
            # in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
            self.conv0_0 = convbn(3, base_filter * 2, kernel_size=3, stride=1, dilation=1)
            self.conv0_1 = convbn(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
            self.conv0_2 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=2)
            self.conv0_3 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv1_2 with conv0_2
            self.conv1_1 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=3)
            self.conv1_2 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            # conv2_2 with conv0_2
            self.conv2_1 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=4)
            self.conv2_2 = convbn(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

            self.conv = nn.Conv2d(base_filter * 12, base_filter*4, 3, 1, 1)

    def forward(self, x):
    
        x=self.conv_2(self.conv_1(self.conv_0(x)))
        conv0_0 = self.conv0_0(x)
        conv0_1 = self.conv0_1(conv0_0)
        conv0_2 = self.conv0_2(conv0_1)
        conv0_3 = self.conv0_3(conv0_2)

        conv1_2 = self.conv1_2(self.conv1_1(conv0_2))

        conv2_2 = self.conv2_2(self.conv2_1(conv0_2))

        conv = self.conv(torch.cat([conv0_3, conv1_2, conv2_2], 1))
        return conv

# class FeatNetGN(nn.Module):
#     def __init__(self):
#         super(FeatNetGN, self).__init__()
#         base_filter = 8
#         # in_channels, out_channels, kernel_size=3, stride=1, pad=1
#         # in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
#         self.conv0_0 = convgnrelu(3, base_filter * 2, kernel_size=3, stride=1, dilation=1)
#         self.conv0_1 = convgnrelu(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
#         self.conv0_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=2)
#         self.conv0_3 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

#         # conv1_2 with conv0_2
#         self.conv1_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=3)
#         self.conv1_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

#         # conv2_2 with conv0_2
#         self.conv2_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=4)
#         self.conv2_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

#         # with concat conv0_3, conv1_2, conv2_2
#         self.conv = nn.Conv2d(base_filter * 12, base_filter*4, 3, 1, 1)

#     def forward(self, x):
        
#         conv0_0 = self.conv0_0(x)
#         conv0_1 = self.conv0_1(conv0_0)
#         conv0_2 = self.conv0_2(conv0_1)
#         conv0_3 = self.conv0_3(conv0_2)

#         conv1_2 = self.conv1_2(self.conv1_1(conv0_2))

#         conv2_2 = self.conv2_2(self.conv2_1(conv0_2))

#         conv = self.conv(torch.cat([conv0_3, conv1_2, conv2_2], 1))
#         return conv



# input 3D Feature Volume
class UNetConvLSTM(nn.Module): # input 3D feature volume
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, gn=True):
        super(UNetConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size #feature: height, width)
        self.gn = gn
        print('Training Phase in UNetConvLSTM: {}, {}, gn: {}'.format(self.height, self.width, self.gn))
        self.input_dim  = input_dim # input channel
        self.hidden_dim = hidden_dim # output channel [16, 16, 16, 16, 16, 8]
        self.kernel_size = kernel_size # kernel size  [[3, 3]*5]
        self.num_layers = num_layers # Unet layer size: must be odd
        self.batch_first = batch_first # TRUE
        self.bias = bias #
        self.return_all_layers = return_all_layers

        cell_list = []
        #assert self.num_layers % 2  == 1 # Even
        self.down_num = (self.num_layers+1) / 2 

        # use GN 
        if self.gn:
            for i in range(0, self.num_layers):
                #cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
                scale = 2**i if i < self.down_num else 2**(self.num_layers-i-1)
                #cell_list.append(ConvGnLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                cell_list.append(ConvLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                            input_dim=self.input_dim[i],
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

            self.cell_list = nn.ModuleList(cell_list)
            self.deconv_0 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            self.deconv_1 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)
        else:
            for i in range(0, self.num_layers):
                #cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
                scale = 2**i if i < self.down_num else 2**(self.num_layers-i-1)
                cell_list.append(ConvBnLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                            input_dim=self.input_dim[i],
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

            self.cell_list = nn.ModuleList(cell_list)
            self.deconv_0 = deConvBnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                output_padding=1
                )
            self.deconv_1 = deConvBnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                output_padding=1
                )
            self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)

    def forward(self, input_tensor, hidden_state=None, idx = 0, process_sq=True):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if idx ==0 : # input the first layer of input image
           hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        if process_sq:
            
            #print(torch.sum(self.hidden_state[0][0]==0))
            h0, c0 = hidden_state[0]= self.cell_list[0](input_tensor=cur_layer_input,
                                                cur_state=hidden_state[0])
            #self.hidden_state[0] = (h0, c0)

            h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
            h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1, 
                                                cur_state=hidden_state[1])
            #self.hidden_state[1] = (h1, c1)
            h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
            h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0,
                                                cur_state=hidden_state[2])
            #self.hidden_state[2] = (h2, c2)
            h2_0 = self.deconv_0(h2) # auto reuse

            h2_1 = torch.cat([h2_0, h1], 1)
            h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_1,
                                                cur_state=hidden_state[3])
            #self.hidden_state[3] = (h3, c3)
            h3_0 = self.deconv_1(h3) # auto reuse
            h3_1 = torch.cat([h3_0, h0], 1)
            h4, c4 = hidden_state[4] = self.cell_list[4](input_tensor=h3_1,
                                                cur_state=hidden_state[4])
            #self.hidden_state[4] = (h4, c4)
            
            cost = self.conv_0(h4) # auto reuse
            #cost = F.tanh(cost)
            # output cost
            return cost, hidden_state
        else:   
            for t in range(seq_len):
                #if t == 0:
                h0, c0 = self.cell_list[0](input_tensor=cur_layer_input[:, t, :, :, :],
                                                    cur_state=hidden_state[0])
                hidden_state[0] = [h0, c0]
                h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
                h1, c1 = self.cell_list[1](input_tensor=h0_1, 
                                                    cur_state=hidden_state[1])
                hidden_state[1] = [h1, c1]
                h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
                h2, c2 = self.cell_list[2](input_tensor=h1_0,
                                                    cur_state=hidden_state[2])
                hidden_state[2] = [h2, c2]
                h2_0 = self.deconv_0(h2) # auto reuse

                h2_1 = torch.concat([h2_0, h1], 1)
                h3, c3 = self.cell_list[3](input_tensor=h2_1,
                                                    cur_state=hidden_state[3])
                hidden_state[3] = [h3, c3]
                h3_0 = self.deconv_1(h3) # auto reuse
                h3_1 = torch.concat([h3_0, h0], 1)
                h4, c4 = self.cell_list[4](input_tensor=h3_1,
                                                    cur_state=hidden_state[4])
                hidden_state[4] = [h4, c4]
                
                cost = self.conv_0(h4) # auto reuse
                cost = nn.Tanh(cost)
                # output cost
                layer_output_list.append(cost)

            prob_volume = torch.stack(layer_output_list, dim=1)
            #prob_volume = nn.Softmax(prob_volume, dim=1)

            return prob_volume

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class UNetConvLSTMV4(UNetConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, gn=True):
        super(UNetConvLSTMV4, self).__init__(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                batch_first, bias, return_all_layers, gn)
        # redefine four UNet-LSTM
        # GPU Memory: 8033MiB
        cell_list = []
        assert self.num_layers % 2  == 1 # num_layers == 7
        self.down_num = (self.num_layers+1) / 2 

        # use GN 
        if self.gn:
            for i in range(0, self.num_layers):
                #cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
                scale = 2**i if i < self.down_num else 2**(self.num_layers-i-1)
                #cell_list.append(ConvGnLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                cell_list.append(ConvLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                            input_dim=self.input_dim[i],
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

            self.cell_list = nn.ModuleList(cell_list)
            self.deconv_0 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            self.deconv_1 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            # add one more deeper network
            self.deconv_2 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)
    
    def forward(self, input_tensor, hidden_state=None, idx = 0, process_sq=True):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if idx ==0 : # input the first layer of input image
           hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        if process_sq:
            #Encoder
            #print(torch.sum(self.hidden_state[0][0]==0))
            h0, c0 = hidden_state[0]= self.cell_list[0](input_tensor=cur_layer_input,
                                                cur_state=hidden_state[0])
            #self.hidden_state[0] = (h0, c0)

            h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
            h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1, 
                                                cur_state=hidden_state[1])
            #self.hidden_state[1] = (h1, c1)
            h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
            h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0,
                                                cur_state=hidden_state[2])

            h2_0 = nn.MaxPool2d((2, 2), stride=2)(h2)  
            h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_0,
                                                cur_state=hidden_state[3])

            # Decoder
            #self.hidden_state[2] = (h2, c2)
            h3_0 = self.deconv_0(h3) # auto reuse
            h3_1 = torch.cat([h3_0, h2], 1)
            h4, c4 = hidden_state[4] = self.cell_list[4](input_tensor=h3_1,
                                                cur_state=hidden_state[4])
            #self.hidden_state[3] = (h3, c3)
            h4_0 = self.deconv_1(h4) # auto reuse
            h4_1 = torch.cat([h4_0, h1], 1)
            h5, c5 = hidden_state[5] = self.cell_list[5](input_tensor=h4_1,
                                                cur_state=hidden_state[5])
        
            #self.hidden_state[3] = (h3, c3)
            h5_0 = self.deconv_2(h5) # auto reuse
            h5_1 = torch.cat([h5_0, h0], 1)
            h6, c6 = hidden_state[6] = self.cell_list[6](input_tensor=h5_1,
                                                cur_state=hidden_state[6])
            #self.hidden_state[4] = (h4, c4)
            
            cost = self.conv_0(h6) # auto reuse
            #cost = F.tanh(cost)
            # output cost
            return cost, hidden_state

class UNetPPConvLSTMV3(UNetConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, gn=True):
        super(UNetPPConvLSTMV3, self).__init__(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                batch_first, bias, return_all_layers, gn)
        # redefine four UNet-LSTM
        cell_list = []
        assert self.num_layers % 2  == 0 # num_layers == 7
        self.down_num = (self.num_layers) / 2 

        # use GN 
        if self.gn:
            scales = [1, 2, 4, 2, 1, 1]
            for i in range(0, self.num_layers):
                #cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
                scale = scales[i]
                #cell_list.append(ConvGnLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                cell_list.append(ConvLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                            input_dim=self.input_dim[i],
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

            self.cell_list = nn.ModuleList(cell_list)
            self.deconv_0 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            self.deconv_1 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            # add one more deeper network
            self.deconv_2 = deConvGnReLU(
                16,
                16, #16
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                output_padding=1
                )
            self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)
    
    def forward(self, input_tensor, hidden_state=None, idx = 0, process_sq=True):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if idx ==0 : # input the first layer of input image
           hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        if process_sq:
            #Encoder
            #print(torch.sum(self.hidden_state[0][0]==0))
            h0, c0 = hidden_state[0]= self.cell_list[0](input_tensor=cur_layer_input,
                                                cur_state=hidden_state[0])
            #self.hidden_state[0] = (h0, c0)

            h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
            h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1, 
                                                cur_state=hidden_state[1])
            #self.hidden_state[1] = (h1, c1)
            h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
            h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0,
                                                cur_state=hidden_state[2])

            # Decoder
            #self.hidden_state[2] = (h2, c2)
            h2_0 = self.deconv_0(h2) # auto reuse

            h2_1 = torch.cat([h2_0, h1], 1)
            h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_1,
                                                cur_state=hidden_state[3])
            
            #self.hidden_state[3] = (h3, c3)
            h1_1 = self.deconv_1(h1) # extra mid layer
            h1_2 = torch.cat([h1_1, h0], 1)
            h11, c11 = hidden_state[4] = self.cell_list[4](input_tensor=h1_2,
                                                cur_state=hidden_state[4])
            #self.hidden_state[4] = (h4, c4)
            
            #self.hidden_state[3] = (h3, c3)
            h3_0 = self.deconv_2(h3) # auto reuse
            h3_1 = torch.cat([h3_0, h0, h11], 1)
            h4, c4 = hidden_state[5] = self.cell_list[5](input_tensor=h3_1,
                                                cur_state=hidden_state[5])

            cost = self.conv_0(h4) # auto reuse
            #cost = F.tanh(cost)
            # output cost
            return cost, hidden_state


class UNetPPConvLSTMV3UPS(UNetConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, gn=True):
        super(UNetPPConvLSTMV3UPS, self).__init__(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                batch_first, bias, return_all_layers, gn)
        # redefine four UNet-LSTM
        cell_list = []
        assert self.num_layers % 2  == 0 # num_layers == 7
        self.down_num = (self.num_layers) / 2 

        # use GN 
        if self.gn:
            scales = [1, 2, 4, 2, 1, 1]
            for i in range(0, self.num_layers):
                #cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
                scale = scales[i]
                #cell_list.append(ConvGnLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                cell_list.append(ConvLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                            input_dim=self.input_dim[i],
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

            self.cell_list = nn.ModuleList(cell_list)
            self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)
            self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            del self.deconv_0
            del self.deconv_1
    
    def forward(self, input_tensor, hidden_state=None, idx = 0, process_sq=True):
        """
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if idx ==0 : # input the first layer of input image
           hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        if process_sq:
            #Encoder
            #print(torch.sum(self.hidden_state[0][0]==0))
            h0, c0 = hidden_state[0]= self.cell_list[0](input_tensor=cur_layer_input,
                                                cur_state=hidden_state[0])
            #self.hidden_state[0] = (h0, c0)

            h0_1 = self.pool(h0)
            h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1, 
                                                cur_state=hidden_state[1])
            #self.hidden_state[1] = (h1, c1)
            h1_0 = self.pool(h1)  
            h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0,
                                                cur_state=hidden_state[2])

            # Decoder
            #self.hidden_state[2] = (h2, c2)
            h2_0 = self.Up(h2) # auto reuse

            h2_1 = torch.cat([h2_0, h1], 1)
            h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_1,
                                                cur_state=hidden_state[3])
            
            #self.hidden_state[3] = (h3, c3)
            h1_1 = self.Up(h1) # extra mid layer
            h1_2 = torch.cat([h1_1, h0], 1)
            h11, c11 = hidden_state[4] = self.cell_list[4](input_tensor=h1_2,
                                                cur_state=hidden_state[4])
            #self.hidden_state[4] = (h4, c4)
            
            #self.hidden_state[3] = (h3, c3)
            h3_0 = self.Up(h3) # auto reuse
            h3_1 = torch.cat([h3_0, h0, h11], 1)
            h4, c4 = hidden_state[5] = self.cell_list[5](input_tensor=h3_1,
                                                cur_state=hidden_state[5])

            cost = self.conv_0(h4) # auto reuse
            #cost = F.tanh(cost)
            # output cost
            return cost, hidden_state


class CostAggNet(nn.Module): 
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, bias=True, gn=True):
        super(CostAggNet, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim # 32, 8
        self.hidden_dim = hidden_dim # 8, 1
        self.kernel_size = kernel_size # 2 * (3, 3)
        self.num_layers = num_layers
        self.bias = bias
        
        gatenet = []
        for i in range(0, self.num_layers):
            #gatenet.append(ConvGnLSTMCell(input_size=(self.height, self.width),
            gatenet.append(ConvLSTMCell(input_size=(self.height, self.width),
                                            input_dim=self.input_dim[i],
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))
        self.gatenet = nn.ModuleList(gatenet)
        self.conv = nn.Conv2d(hidden_dim[-1], 1, 1, stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, hidden_state=None):

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        for i in range(0, self.num_layers):
            hidden_state[i]= self.gatenet[i](input_tensor=input_tensor,
                                            cur_state=hidden_state[i])
        
        h_out, c_out = hidden_state[-1]
        reweight = self.sigmoid(self.conv(h_out))
        return reweight, hidden_state

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.gatenet[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param




