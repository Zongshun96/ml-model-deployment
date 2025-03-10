from utils.config import DEVICE
import time

def sdn_test_early_exits(model, b_x):
    IaaS_runtime_0 = time.time()
    b_x = b_x.to(DEVICE)
    # high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, local_time_l, remote_prep_time_l, remote_time_l, remote_finish_time_l, FaaS_metrics_d_l = [],[],[],[],[],[],[],[],[],[]
    high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, IaaS_metrics_d_l, FaaS_metrics_d_l = [],[],[],[],[],[],[]
    for counter_metrics, progress_metrics in model(b_x):
        (high_conf_mask, batch_output, batch_output_id, processed_output_id, is_early, local_time, layer_time, self_layer_time, self_output_time, remote_prep_time, remote_time, remote_finish_time, FaaS_metrics_d), (layer_time_l, self_layer_time_l, self_output_time_l) = counter_metrics, progress_metrics
        IaaS_runtime_1 = time.time()
        high_conf_mask_l.append(high_conf_mask)
        output_l.append(batch_output)
        output_id_l.append(batch_output_id)
        is_early_l.append(is_early)
        processed_output_id_l.append(processed_output_id)
        FaaS_metrics_d_l.append(FaaS_metrics_d)
        IaaS_metrics_d = {}
        IaaS_metrics_d["IaaS_layer_time"] = layer_time
        IaaS_metrics_d["IaaS_local_time"] = local_time
        IaaS_metrics_d["IaaS_self_layer_time"] = self_layer_time
        IaaS_metrics_d["IaaS_self_output_time"] = self_output_time
        IaaS_metrics_d["IaaS_layer_time_l"] = layer_time_l
        IaaS_metrics_d["IaaS_self_layer_time_l"] = self_layer_time_l
        IaaS_metrics_d["IaaS_self_output_time_l"] = self_output_time_l
        IaaS_metrics_d["IaaS_remote_prep_time"] = remote_prep_time
        IaaS_metrics_d["IaaS_remote_time"] = remote_time
        IaaS_metrics_d["IaaS_remote_finish_time"] = remote_finish_time
        IaaS_metrics_d["IaaS_runtime"] = [IaaS_runtime_1-IaaS_runtime_0]*len(batch_output)
        IaaS_metrics_d_l.append(IaaS_metrics_d)

    return high_conf_mask_l, output_l, output_id_l, processed_output_id_l, is_early_l, IaaS_metrics_d_l, FaaS_metrics_d_l