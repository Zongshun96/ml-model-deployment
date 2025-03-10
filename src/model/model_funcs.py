# model_funcs.py
# implements the functions for training, testing SDNs and CNNs
# also implements the functions for computing confusion and confidence

import torch
import math
import copy
import time
import random

import torch.nn as nn
import numpy as np

from torch.optim import SGD
from random import choice, shuffle
from collections import Counter

from model import aux_funcs as af
# import data
# from ptflops import get_model_complexity_info


# def sdn_training_step(optimizer, model, coeffs, batch, device):
#     b_x = batch[0].to(device)
#     b_y = batch[1].to(device)
#     output = model(b_x)
#     optimizer.zero_grad()  #clear gradients for this training step
#     total_loss = 0.0

#     for ic_id in range(model.num_output - 1):
#         cur_output = output[ic_id]
#         cur_loss = float(coeffs[ic_id])*af.get_loss_criterion()(cur_output, b_y)
#         total_loss += cur_loss

#     total_loss += af.get_loss_criterion()(output[-1], b_y)
#     total_loss.backward()
#     optimizer.step()                # apply gradients

#     return total_loss

# def sdn_ic_only_step(optimizer, model, batch, device):
#     b_x = batch[0].to(device)
#     b_y = batch[1].to(device)
#     output = model(b_x)
#     optimizer.zero_grad()  #clear gradients for this training step
#     total_loss = 0.0

#     for output_id, cur_output in enumerate(output):
#         if output_id == model.num_output - 1: # last output
#             break
        
#         cur_loss = af.get_loss_criterion()(cur_output, b_y)
#         total_loss += cur_loss

#     total_loss.backward()
#     optimizer.step()                # apply gradients

#     return total_loss

# def get_loader(data, augment):
#     if augment:
#         train_loader = data.aug_train_loader
#     else:
#         train_loader = data.train_loader

#     return train_loader  


# def sdn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
#     augment = model.augment_training
#     metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
#     # max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]) # max tau_i --- C_i values
#     max_coeffs = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.75, 0.9]) # max tau_i --- C_i values

#     if model.ic_only:
#         print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
#     else:
#         print('sdn will be trained from scratch...(The SDN training)')

#     for epoch in range(1, epochs+1):
#         cur_lr = af.get_lr(optimizer)
#         print('\nEpoch: {}/{}'.format(epoch, epochs))
#         print('Cur lr: {}'.format(cur_lr))

#         if model.ic_only is False:
#             # calculate the IC coeffs for this epoch for the weighted objective function
#             cur_coeffs = 0.01 + epoch*(max_coeffs/epochs) # to calculate the tau at the currect epoch
#             cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
#             print('Cur coeffs: {}'.format(cur_coeffs))

#         start_time = time.time()
#         model.train()
#         loader = get_loader(data, augment)
#         for i, batch in enumerate(loader):
#             if model.ic_only is False:
#                 total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device)
#             else:
#                 total_loss = sdn_ic_only_step(optimizer, model, batch, device)

#             if i % 100 == 0:
#                 print('batch_idx: {}, Loss: {}: '.format(i, total_loss))

#         top1_test, top5_test = sdn_test(model, data.test_loader, device)

#         print('Top1 Test accuracies: {}'.format(top1_test))
#         print('Top5 Test accuracies: {}'.format(top5_test))
#         end_time = time.time()

#         metrics['test_top1_acc'].append(top1_test)
#         metrics['test_top5_acc'].append(top5_test)

#         top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
#         print('Top1 Train accuracies: {}'.format(top1_train))
#         print('Top5 Train accuracies: {}'.format(top5_train))
#         metrics['train_top1_acc'].append(top1_train)
#         metrics['train_top5_acc'].append(top5_train)

#         epoch_time = int(end_time-start_time)
#         metrics['epoch_times'].append(epoch_time)
#         print('Epoch took {} seconds.'.format(epoch_time))

#         metrics['lrs'].append(cur_lr)
#         scheduler.step()

#     return metrics

# def sdn_test(model, loader, device='cpu'):
#     model.eval()
#     top1 = []
#     top5 = []
#     for output_id in range(model.num_output):
#         t1 = data.AverageMeter()
#         t5 = data.AverageMeter()
#         top1.append(t1)
#         top5.append(t5)

#     with torch.no_grad():
#         for batch in loader:
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             output = model(b_x)
#             for output_id in range(model.num_output):
#                 cur_output = output[output_id]
#                 prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
#                 top1[output_id].update(prec1[0], b_x.size(0))
#                 top5[output_id].update(prec5[0], b_x.size(0))


#     top1_accs = []
#     top5_accs = []

#     for output_id in range(model.num_output):
#         top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
#         top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

#     return top1_accs, top5_accs

# def sdn_get_detailed_results(model, loader, device='cpu'):
#     model.eval()
#     layer_correct = {}
#     layer_wrong = {}
#     layer_predictions = {}
#     layer_confidence = {}

#     outputs = list(range(model.num_output))

#     for output_id in outputs:
#         layer_correct[output_id] = set()
#         layer_wrong[output_id] = set()
#         layer_predictions[output_id] = {}
#         layer_confidence[output_id] = {}

#     with torch.no_grad():
#         for cur_batch_id, batch in enumerate(loader):
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             output = model(b_x)
#             output_sm = [nn.functional.softmax(out, dim=1) for out in output]
#             for output_id in outputs:
#                 cur_output = output[output_id]
#                 cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

#                 pred = cur_output.max(1, keepdim=True)[1]
#                 is_correct = pred.eq(b_y.view_as(pred))
#                 for test_id in range(len(b_x)):
#                     cur_instance_id = test_id + cur_batch_id*loader.batch_size
#                     correct = is_correct[test_id]
#                     layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
#                     layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
#                     if correct == 1:
#                         layer_correct[output_id].add(cur_instance_id)
#                     else:
#                         layer_wrong[output_id].add(cur_instance_id)

#     return layer_correct, layer_wrong, layer_predictions, layer_confidence


# def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
#     model.eval()
#     layer_correct = {}
#     layer_wrong = {}
#     instance_confusion = {}
#     outputs = list(range(model.num_output))

#     for output_id in outputs:
#         layer_correct[output_id] = set()
#         layer_wrong[output_id] = set()

#     with torch.no_grad():
#         for cur_batch_id, batch in enumerate(loader):
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             output = model(b_x)
#             output = [nn.functional.softmax(out, dim=1) for out in output]
#             cur_confusion = af.get_confusion_scores(output, confusion_stats, device)
            
#             for test_id in range(len(b_x)):
#                 cur_instance_id = test_id + cur_batch_id*loader.batch_size
#                 instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
#                 for output_id in outputs:
#                     cur_output = output[output_id]
#                     pred = cur_output.max(1, keepdim=True)[1]
#                     is_correct = pred.eq(b_y.view_as(pred))
#                     correct = is_correct[test_id]
#                     if correct == 1:
#                         layer_correct[output_id].add(cur_instance_id)
#                     else:
#                         layer_wrong[output_id].add(cur_instance_id)

#     return layer_correct, layer_wrong, instance_confusion

# # to normalize the confusion scores
# def sdn_confusion_stats(model, loader, device='cpu'):
#     model.eval()
#     outputs = list(range(model.num_output))
#     confusion_scores = []

#     total_num_instances = 0
#     with torch.no_grad():
#         for batch in loader:
#             b_x = batch[0].to(device)
#             total_num_instances += len(b_x)
#             output = model(b_x)
#             output = [nn.functional.softmax(out, dim=1) for out in output]
#             cur_confusion = af.get_confusion_scores(output, None, device)
#             for test_id in range(len(b_x)):
#                 confusion_scores.append(cur_confusion[test_id].cpu().numpy())

#     confusion_scores = np.array(confusion_scores)
#     mean_con = float(np.mean(confusion_scores))
#     std_con = float(np.std(confusion_scores))
#     return (mean_con, std_con)

# def sdn_test_early_exits(model, loader, max_batch_count=2, device='cpu'):
#     model.eval()
#     # early_output_counts = [0] * model.num_output
#     # non_conf_output_counts = [0] * model.num_output
#     output_counts = [0] * model.num_output
#     processed_output_counts = [0] * model.num_output

#     top1 = data.AverageMeter()
#     top5 = data.AverageMeter()
#     # total_time, total_local_time, total_remote_prep_time, total_remote_time, total_remote_finish_time = 0, 0, 0, 0, 0
#     # total_time, total_local_time, total_remote_prep_time, total_remote_time, total_remote_finish_time, total_FaaS_metrics_d = [],[],[],[],[],[]
#     total_IaaS_metrics_d, total_FaaS_metrics_d = [],[]
#     with torch.no_grad():
#         batch_count = 0
#         # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
#         # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#         for batch in loader:
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             # # ============================
#             # from torch.profiler import profile, record_function, ProfilerActivity
#             # with torch.no_grad():
#             #     # for _ in range(5):
#             #     # inputs = torch.randn(1, 3, 224, 224).to(device)
#             #     with profile(activities=[
#             #                 ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#             #             with record_function("model_inference"):
#             #                 output = model(b_x)
#             #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#             #     # Profile the model
#             #     with profile(activities=[ProfilerActivity.CPU], with_flops=True, with_modules=True) as prof:
#             #         model(b_x)
#             #         # Print the profiling results
#             #         print(prof.key_averages(group_by_input_shape=True).table(sort_by="flops", row_limit=10))
#             # # ============================
#             # # ============================
#             # import profiler
#             # print(profiler.profile_sdn(model, 32, "cpu"))
#             # # ============================
#             # # ============================
#             # from ptflops import get_model_complexity_info
#             # macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
#             # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#             # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#             # # ============================
#             # # # ============================
#             # from torchsummary import summary
#             # summary(model, (3, 32, 32))
#             # # # ============================
#             IaaS_runtime_0 = time.time()
#             # high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, local_time_l, remote_prep_time_l, remote_time_l, remote_finish_time_l, FaaS_metrics_d_l = [],[],[],[],[],[],[],[],[],[]
#             high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, IaaS_metrics_d_l, FaaS_metrics_d_l = [],[],[],[],[],[],[]
#             for counter_metrics, progress_metrics in model(b_x):
#                 (high_conf_mask, batch_output, batch_output_id, processed_output_id, is_early, local_time, layer_time, self_layer_time, self_output_time, remote_prep_time, remote_time, remote_finish_time, FaaS_metrics_d), (layer_time_l, self_layer_time_l, self_output_time_l) = counter_metrics, progress_metrics
#                 IaaS_runtime_1 = time.time()
#                 high_conf_mask_l.append(high_conf_mask)
#                 output_l.append(batch_output)
#                 output_id_l.append(batch_output_id)
#                 is_early_l.append(is_early)
#                 processed_output_id_l.append(processed_output_id)
#                 FaaS_metrics_d_l.append(FaaS_metrics_d)
#                 IaaS_metrics_d = {}
#                 IaaS_metrics_d["IaaS_layer_time"] = layer_time
#                 IaaS_metrics_d["IaaS_local_time"] = local_time
#                 IaaS_metrics_d["IaaS_self_layer_time"] = self_layer_time
#                 IaaS_metrics_d["IaaS_self_output_time"] = self_output_time
#                 IaaS_metrics_d["IaaS_layer_time_l"] = layer_time_l
#                 IaaS_metrics_d["IaaS_self_layer_time_l"] = self_layer_time_l
#                 IaaS_metrics_d["IaaS_self_output_time_l"] = self_output_time_l
#                 IaaS_metrics_d["IaaS_remote_prep_time"] = remote_prep_time
#                 IaaS_metrics_d["IaaS_remote_time"] = remote_time
#                 IaaS_metrics_d["IaaS_remote_finish_time"] = remote_finish_time
#                 IaaS_metrics_d["IaaS_runtime"] = [IaaS_runtime_1-IaaS_runtime_0]*len(batch_output)
#                 IaaS_metrics_d_l.append(IaaS_metrics_d)

#             # y_idx = 0
#             # for high_conf_mask, batch_output, batch_output_id, batch_processed_output_id, is_early, local_time, remote_prep_time, remote_time, remote_finish_time, FaaS_metrics_d in zip(high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, local_time_l, remote_prep_time_l, remote_time_l, remote_finish_time_l, FaaS_metrics_d_l):
#             for high_conf_mask, batch_output, batch_output_id, batch_processed_output_id, is_early, IaaS_metrics_d, FaaS_metrics_d in zip(high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, IaaS_metrics_d_l, FaaS_metrics_d_l):
#                 total_IaaS_metrics_d.append(IaaS_metrics_d)
#                 total_FaaS_metrics_d.append(FaaS_metrics_d)

#                 for output_id in batch_output_id:
#                     output_counts[output_id] += 1
#                 for processed_output_id in batch_processed_output_id:
#                     processed_output_counts[processed_output_id] += 1

#                 # prec1, prec5 = data.accuracy(batch_output, b_y[y_idx:y_idx+len(batch_output)], topk=(1, 5))
#                 prec1, prec5 = data.accuracy(batch_output, b_y[high_conf_mask], topk=(1, 5))
#                 b_y = b_y[~high_conf_mask]
#                 # y_idx += len(batch_output)
#                 top1.update(prec1[0], len(batch_output))
#                 top5.update(prec5[0], len(batch_output))

#             batch_count += 1
#             if batch_count >= max_batch_count:
#                 break

#     top1_acc = top1.avg.data.cpu().numpy()[()]
#     top5_acc = top5.avg.data.cpu().numpy()[()]

#     return top1_acc, top5_acc, output_counts, processed_output_counts, total_IaaS_metrics_d, total_FaaS_metrics_d

# def cnn_training_step(model, optimizer, data, labels, device='cpu'):
#     b_x = data.to(device)   # batch x
#     b_y = labels.to(device)   # batch y
#     output = model(b_x)            # cnn final output
#     criterion = af.get_loss_criterion()
#     loss = criterion(output, b_y)   # cross entropy loss
#     optimizer.zero_grad()           # clear gradients for this training step
#     loss.backward()                 # backpropagation, compute gradients
#     optimizer.step()                # apply gradients


# def cnn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
#     metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

#     for epoch in range(1, epochs+1):
#         cur_lr = af.get_lr(optimizer)

#         if not hasattr(model, 'augment_training') or model.augment_training:
#             train_loader = data.aug_train_loader
#         else:
#             train_loader = data.train_loader

#         start_time = time.time()
#         model.train()
#         print('Epoch: {}/{}'.format(epoch, epochs))
#         print('Cur lr: {}'.format(cur_lr))
#         for x, y in train_loader:
#             cnn_training_step(model, optimizer, x, y, device)
        
#         end_time = time.time()
    
#         top1_test, top5_test = cnn_test(model, data.test_loader, device)
#         print('Top1 Test accuracy: {}'.format(top1_test))
#         print('Top5 Test accuracy: {}'.format(top5_test))
#         metrics['test_top1_acc'].append(top1_test)
#         metrics['test_top5_acc'].append(top5_test)

#         top1_train, top5_train = cnn_test(model, train_loader, device)
#         print('Top1 Train accuracy: {}'.format(top1_train))
#         print('Top5 Train accuracy: {}'.format(top5_train))
#         metrics['train_top1_acc'].append(top1_train)
#         metrics['train_top5_acc'].append(top5_train)
#         epoch_time = int(end_time-start_time)
#         print('Epoch took {} seconds.'.format(epoch_time))
#         metrics['epoch_times'].append(epoch_time)

#         metrics['lrs'].append(cur_lr)
#         scheduler.step()

#     return metrics
    

# def cnn_test_time(model, loader, device='cpu'):
#     model.eval()
#     top1 = data.AverageMeter()
#     top5 = data.AverageMeter()
#     total_time = 0
#     with torch.no_grad():
#         for batch in loader:
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             start_time = time.time()
#             output = model(b_x)
#             end_time = time.time()
#             total_time += (end_time - start_time)
#             prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
#             top1.update(prec1[0], b_x.size(0))
#             top5.update(prec5[0], b_x.size(0))

#     top1_acc = top1.avg.data.cpu().numpy()[()]
#     top5_acc = top5.avg.data.cpu().numpy()[()]

#     return top1_acc, top5_acc, total_time


# def cnn_test(model, loader, device='cpu'):
#     model.eval()
#     top1 = data.AverageMeter()
#     top5 = data.AverageMeter()

#     with torch.no_grad():
#         for batch in loader:
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             output = model(b_x)
#             prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
#             top1.update(prec1[0], b_x.size(0))
#             top5.update(prec5[0], b_x.size(0))

#     top1_acc = top1.avg.data.cpu().numpy()[()]
#     top5_acc = top5.avg.data.cpu().numpy()[()]

#     return top1_acc, top5_acc


# def cnn_get_confidence(model, loader, device='cpu'):
#     model.eval()
#     correct = set()
#     wrong = set()
#     instance_confidence = {}
#     correct_cnt = 0

#     with torch.no_grad():
#         for cur_batch_id, batch in enumerate(loader):
#             b_x = batch[0].to(device)
#             b_y = batch[1].to(device)
#             output = model(b_x)
#             output = nn.functional.softmax(output, dim=1)
#             model_pred = output.max(1, keepdim=True)
#             pred = model_pred[1].to(device)
#             pred_prob = model_pred[0].to(device)

#             is_correct = pred.eq(b_y.view_as(pred))
#             correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

#             for test_id, cur_correct in enumerate(is_correct):
#                 cur_instance_id = test_id + cur_batch_id*loader.batch_size
#                 instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

#                 if cur_correct == 1:
#                     correct.add(cur_instance_id)
#                 else:
#                     wrong.add(cur_instance_id)

   
#     return correct, wrong, instance_confidence