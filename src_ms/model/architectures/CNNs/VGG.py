import torch
import math

import torch.nn as nn
import torchvision.models as models

import aux_funcs as af
import model_funcs as mf


class ConvBlock(nn.Module):
    def __init__(self, conv_params, output_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        batch_norm = conv_params[3]
        input_size = output_params[0]

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))
                
        conv_layers.append(nn.ReLU())
                
        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))
        elif max_pool_size == -2:
            conv_layers.append(nn.AdaptiveAvgPool2d((input_size,input_size)))
        
        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(af.Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))        
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class VGG(nn.Module):
    def __init__(self, params):
        super(VGG, self).__init__()
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels'] # the first element is input dimension
        self.fc_layer_sizes = params['fc_layers']

        # read or assign defaults to the rest
        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.use_pretrained = params['use_pretrained']
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test
        self.num_output = 1

        self.init_conv = nn.Sequential() # just for compatibility with other models

        self.layers = nn.ModuleList()
        # add conv layers
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            if layer_id == len(self.conv_channels)-1:
                conv_params =  (input_channel, channel, -2, self.conv_batch_norm)
            else:
                conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            output_params = (cur_input_size,)
            self.layers.append(ConvBlock(conv_params, output_params))
            input_channel = channel
        
        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True
            
            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width
        
        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            if not self.use_pretrained:
                self.initialize_weights()
            else:
                self.initialize_pretrained()


    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def initialize_pretrained(self):
        with torch.no_grad():
            pretrained_model = list(models.vgg16(weights='IMAGENET1K_V1').modules())
            j_idx = 0
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    while not isinstance(pretrained_model[j_idx], nn.Conv2d):
                        j_idx += 1
                    m.weight.copy_(pretrained_model[j_idx].weight)
                    if m.bias is not None:
                        m.bias.copy_(pretrained_model[j_idx].bias)
                    j_idx += 1
                    # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    # m.weight.data.normal_(0, math.sqrt(2. / n))
                    # if m.bias is not None:
                    #     m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()