from collections import defaultdict
import torch
import math

import torch.nn as nn
import numpy as np
import requests

from model import aux_funcs as af
# from model import model_funcs as mf

import zlib, gzip
import lz4.frame
import base64
import json
import boto3
import time
import sys
import io

import logging
import os

# Set the path for the log file
log_file_path = './logs/logfile.log'  # Replace with your desired log file path

# Ensure the directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)

def compress_value(value):
    # Serialize the value to a JSON-formatted string
    json_str = json.dumps(value)
    # Compress the JSON string
    compressed_data = zlib.compress(json_str.encode('utf-8'))
    # Encode the compressed data to a base64 string
    return base64.b64encode(compressed_data).decode('utf-8')

def decompress_value(compressed_value):
    # Decode the base64 string to compressed bytes
    compressed_data = base64.b64decode(compressed_value.encode('utf-8'))
    # Decompress the data
    json_str = zlib.decompress(compressed_data).decode('utf-8')
    # Deserialize the JSON string back to the original value
    return json.loads(json_str)


class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        batch_norm = conv_params[3]
        
        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.depth = 1


        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1, stride=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))
                
        conv_layers.append(nn.ReLU())
                
        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))
        elif max_pool_size == -2:
            conv_layers.append(nn.AdaptiveAvgPool2d((input_size,input_size)))

        self.layers = nn.Sequential(*conv_layers)


        if add_output:
            self.output = af.InternalClassifier(input_size, output_channels, num_classes) 
            self.no_output = False

        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True
        

    def forward(self, x):
        t0 = time.time()
        fwd = self.layers(x)
        t1 = time.time()
        logging.info("running time of self.layers(x): {}".format(t1-t0))
        t2 = time.time()
        output = self.output(fwd)
        t3 = time.time()
        logging.info("running time of self.output(x): {}".format(t3-t2))
        return fwd, 1, output, t1-t0, t3-t2

    def only_output(self, x):
        t0 = time.time()
        fwd = self.layers(x)
        t1 = time.time()
        logging.info("running time of self.layers(x): {}".format(t1-t0))
        t2 = time.time()
        output = self.output(fwd)
        t3 = time.time()
        logging.info("running time of self.output(x): {}".format(t3-t2))
        return None, 1, output, t1-t0, t3-t2

    def only_forward(self, x):
        t0 = time.time()
        fwd = self.layers(x)
        t1 = time.time()
        logging.info("running time of self.layers(x): {}".format(t1-t0))
        return fwd, 0, None, t1-t0, 0

class FcBlockWOutput(nn.Module):
    def __init__(self, fc_params, output_params, flatten=False):
        super(FcBlockWOutput, self).__init__()
        input_size = fc_params[0]
        output_size = fc_params[1]
        
        add_output = output_params[0]
        num_classes = output_params[1]
        self.output_id = output_params[2]
        self.depth = 1

        fc_layers = []

        if flatten:
            fc_layers.append(af.Flatten())

        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))        
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = nn.Linear(output_size, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        t0 = time.time()
        fwd = self.layers(x)
        t1 = time.time()
        logging.info("running time of self.layers(x): {}".format(t1-t0))
        t2 = time.time()
        output = self.output(fwd)
        t3 = time.time()
        logging.info("running time of self.output(x): {}".format(t3-t2))
        return fwd, 1, output, t1-t0, t3-t2

    def only_output(self, x):
        t0 = time.time()
        fwd = self.layers(x)
        t1 = time.time()
        logging.info("running time of self.layers(x): {}".format(t1-t0))
        t2 = time.time()
        output = self.output(fwd)
        t3 = time.time()
        logging.info("running time of self.output(x): {}".format(t3-t2))
        return None, 1, output, t1-t0, t3-t2

    def only_forward(self, x):
        t0 = time.time()
        fwd = self.layers(x)
        t1 = time.time()
        logging.info("running time of self.layers(x): {}".format(t1-t0))
        return fwd, 0, None, t1-t0, 0

class VGG_SDN(nn.Module):
    def __init__(self, params):
        super(VGG_SDN, self).__init__()
        self.params = params
        # self.db = redis.StrictRedis(host="redis")
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
        self.add_output = params['add_ic']
        self.cut_output_idx = -1
        # self.cut_output_idx = params['cut_output_idx']

        # self.train_func = mf.sdn_train
        # self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1

        self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        # self.layers_FaaS = nn.ModuleList() # The second partition which will be in openwhisk.
        self.init_depth = 0
        self.end_depth = 2

        # add conv layers
        input_channel = 3
        cur_input_size = self.input_size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            if layer_id == len(self.conv_channels)-1:
                conv_params =  (input_channel, channel, -2, self.conv_batch_norm)
            else:
                conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)
            self.layers.append(ConvBlockWOutput(conv_params, output_params))
            input_channel = channel
            output_id += add_output
        
        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            add_output = self.add_output[layer_id + len(self.conv_channels)]
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
            fc_input_size = width
            output_id += add_output
        
        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

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

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        # layers
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        # # layers_FaaS
        # for layer in self.layers_FaaS:  
        #     fwd, is_output, output = layer(fwd)
        #     if is_output:
        #         outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit(self, x, CONFIDENCE_THRESHOLD):
        local_time_start = time.time()
        max_confidences = []
        outputs = []
        # passed_layers_l = []

        fwd = self.init_conv(x)
        output_id = 0
        # logging.info("the cut idx is {}".format(self.cut_output_idx))
        logging.info("====================")
        total_layer_time = total_self_layer_time = total_self_output_time = 0
        layer_time_l, self_layer_time_l, self_output_time_l = [], [], []
        # layers
        for layer_idx, layer in enumerate(self.layers):

            # Cut Control
            if output_id < self.cut_output_idx:
                # logging.info(layer)
                # passed_layers_l.append((layer_idx, output_id))
                
                # # ============================
                logging.info('---------------------------- layer {} ------------------'.format(layer_idx))
                logging.info(layer)
                # ============================
                from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
                # with torch.no_grad():
                #     # Profile the model
                #     with profile(activities=[ProfilerActivity.CPU], with_flops=True, with_modules=True) as prof:
                #         layer(fwd)
                #         # Print the profiling results
                #     logging.info(prof.key_averages(group_by_input_shape=True).table(sort_by="flops", row_limit=10))
                with profile(activities=[ProfilerActivity.CPU], with_flops=True, profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler('./log')) as prof:
                    with record_function("model_inference"):
                        layer(fwd)
                logging.info(prof.key_averages().table())
                # ============================
                # # ============================
                # import profiler
                # logging.info(profiler.profile_sdn(layer, 32, "cpu"))
                # # ============================
                # # ============================
                # from ptflops import get_model_complexity_info
                # macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
                # logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                # logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
                # # ============================
                logging.info('---------------------------------------------------------')

                total_layer_time_0 = time.time()
                fwd, is_output, output, self_layer_time, self_output_time = layer(fwd)
                total_layer_time_1 =  time.time()
                layer_time = total_layer_time_1-total_layer_time_0
                total_layer_time += layer_time
                total_self_layer_time += self_layer_time
                total_self_output_time += self_output_time
                logging.info("fwd size: {}".format(fwd.size()))
                logging.info("layer_time = {}; total_layer_time = {}".format(layer_time, total_layer_time))
                logging.info("running time of self.layers(x): {}, total_self_layer_time: {}".format(self_layer_time, total_self_layer_time))
                logging.info("running time of self.output(x): {}, total_self_output_time: {}".format(self_output_time, total_self_output_time))
                logging.info("interemdiate data size: {}".format(fwd.element_size() * fwd.numel()))
                layer_time_l.append(layer_time)
                self_layer_time_l.append(self_layer_time)
                self_output_time_l.append(self_output_time)

                if is_output:
                    outputs.append(output)
                    softmax = nn.functional.softmax(output, dim=1)
                    
                    max_confidence, max_conf_index = torch.max(softmax, dim=1)
                    max_confidences.append(max_confidence)

                    high_conf_mask = max_confidence >= CONFIDENCE_THRESHOLD
                    low_conf_mask = ~high_conf_mask  # Inverse of high_conf_mask

                    # high_conf_samples_confidences = softmax[high_conf_mask]
                    # low_conf_samples_confidences = softmax[low_conf_mask]

                    num_high_conf = torch.sum(high_conf_mask).item()
                    if num_high_conf != 0:
                        # logging.info("EE: confidences:     ", max_confidences)
                        # logging.info("EE: passed_layers_l: ", passed_layers_l)
                        is_early = True

                        max_confidences_trimed, outputs_trimed = [], []
                        for max_confidence, output in zip(max_confidences, outputs):
                            max_confidences_trimed.append(max_confidence[low_conf_mask])
                            outputs_trimed.append(output[low_conf_mask])
                        max_confidences = max_confidences_trimed
                        outputs = outputs_trimed

                        local_time_end = time.time()

                        if num_high_conf == len(fwd):
                            yield (
                                (
                                    high_conf_mask, 
                                    output[high_conf_mask], 
                                    torch.tensor([output_id]*num_high_conf), 
                                    torch.tensor([output_id]*num_high_conf), 
                                    is_early, 
                                    [local_time_end-local_time_start]*num_high_conf, 
                                    [total_layer_time]*num_high_conf, 
                                    [total_self_layer_time]*num_high_conf, 
                                    [total_self_output_time]*num_high_conf, 
                                    [0]*num_high_conf, 
                                    [0]*num_high_conf, 
                                    [0]*num_high_conf, 
                                    {}
                                ), 
                                (
                                    [layer_time_l]*num_high_conf, 
                                    [self_layer_time_l]*num_high_conf, 
                                    [self_output_time_l]*num_high_conf, 
                                )
                            )
                            return  # Signals the end of the generator
                        else:
                            yield (
                                (
                                    high_conf_mask, 
                                    output[high_conf_mask], 
                                    torch.tensor([output_id]*num_high_conf), 
                                    torch.tensor([output_id]*num_high_conf), 
                                    is_early, 
                                    [local_time_end-local_time_start]*num_high_conf, 
                                    [total_layer_time]*num_high_conf, 
                                    [total_self_layer_time]*num_high_conf, 
                                    [total_self_output_time]*num_high_conf, 
                                    [0]*num_high_conf, 
                                    [0]*num_high_conf, 
                                    [0]*num_high_conf, 
                                    {}
                                ),
                                (
                                    [layer_time_l]*num_high_conf, 
                                    [self_layer_time_l]*num_high_conf, 
                                    [self_output_time_l]*num_high_conf, 
                                )
                            )
                            fwd = fwd[low_conf_mask]
                    
                    output_id += is_output
                    # fwd = fwd[low_conf_mask]
            else:
                # logging.info("EE: confidences:     ", max_confidences)
                # logging.info("EE: passed_layers_l: ", passed_layers_l)
                break
        else:            
            # logging.info("FE: passed_layers_l: ", passed_layers_l)
            logging.info('---------------------------- layer {} ------------------'.format(layer_idx+1))
            logging.info(self.end_layers)
            # ============================
            # from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
            # with torch.no_grad():
            #     # Profile the model
            #     with profile(activities=[ProfilerActivity.CPU], with_flops=True, with_modules=True) as prof:
            #         layer(fwd)
            #         # Print the profiling results
            #     logging.info(prof.key_averages(group_by_input_shape=True).table(sort_by="flops", row_limit=10))
            with profile(activities=[ProfilerActivity.CPU], with_flops=True, profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler('./log')) as prof:
                with record_function("model_inference"):
                    self.end_layers(fwd)
            logging.info(prof.key_averages().table())
            # ============================
            # # ============================
            # from ptflops import get_model_complexity_info
            # macs, params = get_model_complexity_info(self.end_layers, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
            # logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # # ============================
            total_layer_time_0 = time.time()
            output = self.end_layers(fwd)
            total_layer_time_1 = time.time()
            self_layer_time = layer_time = total_layer_time_1-total_layer_time_0
            self_output_time = 0
            total_layer_time += layer_time
            local_time_end = time.time()
            logging.info("layer_time/self_layer_time = {}; total_layer_time = {}".format(layer_time, total_layer_time))
            layer_time_l.append(layer_time)
            self_layer_time_l.append(self_layer_time)
            self_output_time_l.append(self_output_time)

            confidence_time_start = time.time()
            outputs.append(output)
            softmax = nn.functional.softmax(output, dim=1)

            max_confidence, max_conf_index = torch.max(softmax, dim=1)
            max_confidences.append(max_confidence)
            # logging.info("FE: confidences:     ", max_confidences)
            max_confidences_output_id = torch.argmax(torch.stack(max_confidences), dim=0)
            is_early = False
            confidence_time_end = time.time()
            logging.info("confidence_time {}".format(confidence_time_end-confidence_time_start))
            
            formating_time_start = time.time()
            (torch.tensor([True]*len(fwd)), torch.stack([outputs[max_confidence_output_id][sample_id] for sample_id, max_confidence_output_id in enumerate(max_confidences_output_id)]), max_confidences_output_id, torch.tensor([output_id]*len(fwd)), is_early, [local_time_end-local_time_start]*len(fwd), [layer_time]*len(fwd), [0]*len(fwd), [layer_time]*len(fwd), [total_layer_time]*len(fwd), [total_self_layer_time]*len(fwd), [total_self_output_time]*len(fwd), [0]*len(fwd), [0]*len(fwd), [0]*len(fwd), {})
            formating_time_end = time.time()
            logging.info("formating_time {}".format(formating_time_end-formating_time_start))
            yield (
                    (
                    torch.tensor([True]*len(fwd)), 
                    torch.stack([outputs[max_confidence_output_id][sample_id] for sample_id, max_confidence_output_id in enumerate(max_confidences_output_id)]), 
                    max_confidences_output_id, 
                    torch.tensor([output_id]*len(fwd)), 
                    is_early, 
                    [local_time_end-local_time_start]*len(fwd), 
                    [total_layer_time]*len(fwd), 
                    [total_self_layer_time]*len(fwd), 
                    [total_self_output_time]*len(fwd), 
                    [0]*len(fwd), 
                    [0]*len(fwd), 
                    [0]*len(fwd), 
                    {}
                ),
                (
                    [layer_time_l]*len(fwd), 
                    [self_layer_time_l]*len(fwd), 
                    [self_output_time_l]*len(fwd), 
                )
            )
            return
            
        local_time_end = time.time()
        # logging.info("I was forward to FaaS at cut idx {}".format(output_id))

        # layers_FaaS
        remote_prep_time_start = time.time()

        data = {
            "layer_idx":layer_idx, # the first layer to run in FaaS.
            'output_id':output_id, 
            # 'device': "cpu", 
            'fwd':fwd, 
            'confidence_threshold':self.confidence_threshold, 
            'outputs':outputs, 
            'max_confidences':max_confidences,
            }
        
        # Serialize the data using torch.save() into a BytesIO buffer
        buffer = io.BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        
        # t0 = time.time()
        # Compress the serialized data
        # compressed_data = gzip.compress(buffer.read())
        compressed_data = lz4.frame.compress(buffer.read())
        # t1 = time.time()
        
        # Encode to base64 for safe transmission
        encoded_data = base64.b64encode(compressed_data).decode('utf-8')
        payload = {
            'data': encoded_data
        }
        remote_prep_time_end = time.time()
        remote_time_start = remote_prep_time_end

        size = sys.getsizeof(json.dumps(payload))
        logging.info(f"Size of dictionary: {size} bytes")      
        
        # Invoke the Lambda function
        # r = requests.post("http://127.0.0.1:3000/classify_digit", json=payload)
        r = requests.post("https://px0myo9fp3.execute-api.us-east-2.amazonaws.com/Prod/classify_digit/", json=payload)
        # lambda_client = boto3.client('lambda', region_name='us-east-2')
        # r = lambda_client.invoke(
        #     FunctionName='pytorch-lambda-example-InferenceFunction-SjR6pspU4srk',
        #     # FunctionName='test_latency',
        #     InvocationType='RequestResponse',  # Synchronous invocation
        #     Payload=json.dumps({'body':json.dumps(payload)})
        # )

        remote_time_end = time.time()
        # logging.info(r)
        remote_finish_time_start = remote_time_end

        # # Parse the JSON response
        response_json = r.json()
        # response_json = json.loads(json.loads(r['Payload'].read().decode('utf-8')).get('body'))

        # Extract the encoded data
        encoded_data = response_json.get('data')

        # Decode from base64
        compressed_data = base64.b64decode(encoded_data)

        # Decompress using gzip
        # decompressed_data = gzip.decompress(compressed_data)
        decompressed_data = lz4.frame.decompress(compressed_data)

        # Deserialize the data using torch.load() from a BytesIO buffer
        buffer = io.BytesIO(decompressed_data)
        loaded_data = torch.load(buffer)

        high_conf_mask_l = loaded_data["high_conf_mask_l"]
        output_l = loaded_data["output_l"]
        output_id_l = loaded_data["output_id_l"]
        processed_output_id_l = loaded_data["processed_output_id_l"]
        is_early_l = loaded_data["is_early_l"]
        metrics_d_l = loaded_data["metrics_d_l"]

        remote_finish_time_end = time.time()

        for high_conf_mask, batch_output, batch_output_id, processed_output_id, is_early, metrics_d in zip(high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, metrics_d_l):
            yield (
                (
                    high_conf_mask, 
                    batch_output, 
                    batch_output_id, 
                    processed_output_id, 
                    is_early, 
                    [local_time_end-local_time_start]*len(batch_output), 
                    [total_layer_time]*len(batch_output), 
                    [total_self_layer_time]*len(batch_output), 
                    [total_self_output_time]*len(batch_output), 
                    [remote_prep_time_end-remote_prep_time_start]*len(batch_output), 
                    [remote_time_end-remote_time_start]*len(batch_output), 
                    [remote_finish_time_end-remote_finish_time_start]*len(batch_output), 
                    metrics_d
                ),
                (
                    [layer_time_l]*len(batch_output), 
                    [self_layer_time_l]*len(batch_output), 
                    [self_output_time_l]*len(batch_output), 
                )
            )
        else:
            return