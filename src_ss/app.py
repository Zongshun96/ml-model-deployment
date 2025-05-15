from flask import Flask, request, jsonify, Response
import gzip
import os
import pickle
import time
import numpy as np
import xgboost as xgb
import scipy.sparse
import scipy
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, hamming_loss, recall_score
# from tqdm import tqdm  # Optional: remove or uncomment for debugging.
import json

# -----------------------------
# Helper Functions
# -----------------------------
def has_intersection(from_set, to_set):
    for ele in from_set:
        if ele in to_set:
            return True
    return False

def get_intersection(from_set, to_set):
    ret = []
    for ele in from_set:
        if ele in to_set:
            ret.append(ele)
    return ret

def tagsets_to_matrix(inference_flag=True,
                      input_size=None, compact_factor=1,
                      all_tags_l=None, tag_index_mapping=None,
                      all_label_l=None, label_index_mapping=None,
                      tags_by_instance_l=None, labels_by_instance_l=None, tagset_files=None,
                      feature_importance=np.array([])):
    """
    Converts incoming tag dictionaries into a feature matrix and (if not inference) a label matrix.
    Mapping objects for tags (all_tags_l, tag_index_mapping) and labels (all_label_l, label_index_mapping)
    are expected to be preloaded and passed in. If they are not provided, a fallback to generating the mapping
    from the corresponding set is used.
    
    Returns:
      - A list of tagset file identifiers for the instances with recognized features,
      - The feature matrix (as a sparse matrix),
      - The label matrix (if inference_flag is False; otherwise an empty array),
      - The set of instance indexes that were processed,
      - The total count of instances,
      - A dictionary of runtime durations (op_durations).
    """
    op_durations = defaultdict(int)
    
    # --- Feature Matrix Generation ---
    t_gen_mat_0 = time.time()
    t_get_feature_0 = time.time()
    instance_row_list = []
    instance_row_idx_set = []
    used_idxs = np.where(feature_importance > 0)[0].tolist()
    used_tags_set = set([all_tags_l[idx] for idx in used_idxs if idx < len(all_tags_l)])
    t_get_feature_t = time.time()
    op_durations["get_feature"] = t_get_feature_t - t_get_feature_0

    for instance_row_idx, instance_tags_d in enumerate(tags_by_instance_l):
        if input_size is None:
            input_size = len(all_tags_l) // compact_factor
        t_selector_0 = time.time()
        used_instance_tags_list = get_intersection(used_tags_set, instance_tags_d.keys())
        if (feature_importance.size != 0) and not used_instance_tags_list:
            t_selector_t = time.time()
            op_durations["selector"] += t_selector_t - t_selector_0
            continue
        t_selector_t = time.time()
        op_durations["selector"] += t_selector_t - t_selector_0
        t_mat_builder_0 = time.time()
        instance_row = np.zeros(input_size)
        for tag_name in used_instance_tags_list:
            instance_row[tag_index_mapping[tag_name] % input_size] = instance_tags_d[tag_name]
        else:
            # print(instance_row)
            instance_row_idx_set.append(instance_row_idx)
            instance_row_list.append(scipy.sparse.csr_matrix(instance_row))
        t_mat_builder_t = time.time()
        op_durations["mat_builder"] += t_mat_builder_t - t_mat_builder_0

    instance_row_count = instance_row_idx + 1 if tags_by_instance_l else 0

    t_list_to_mat_0 = time.time()
    if instance_row_list:
        feature_matrix = scipy.sparse.vstack(instance_row_list)
    else:
        feature_matrix = scipy.sparse.csr_matrix([])
    t_list_to_mat_t = time.time()
    t_gen_mat_t = time.time()
    op_durations["list_to_mat"] = t_list_to_mat_t - t_list_to_mat_0
    op_durations["gen_mat"] = t_gen_mat_t - t_gen_mat_0

    # --- Label Matrix Generation (only if not inference) ---
    label_matrix = np.array([])
    if not inference_flag:
        removed_label_l = []
        instance_row_list_lbl = []
        for instance_row_idx, labels in enumerate(labels_by_instance_l):
            instance_row = np.zeros(len(all_label_l))
            for label in labels:
                if label in label_index_mapping:
                    instance_row[label_index_mapping[label]] = 1
                else:
                    removed_label_l.append(label)
            else:
                instance_row_list_lbl.append(instance_row)
        if instance_row_list_lbl:
            label_matrix = np.vstack(instance_row_list_lbl)
        else:
            label_matrix = np.array([])
        # Optionally: save removed labels if needed.
    return ([tagset_files[instance_row_idx] for instance_row_idx in instance_row_idx_set],
            feature_matrix, label_matrix, instance_row_idx_set, instance_row_count, dict(op_durations))


def one_hot_to_names(mapping_path, one_hot_matrix, mapping=None):
    if mapping is None:
        with open(mapping_path, 'rb') as fp:
            mapping = pickle.load(fp)
    idxs_yx = np.nonzero(one_hot_matrix)
    labels = defaultdict(list)
    for row_idx, col_idx in zip(idxs_yx[0], idxs_yx[1]):
        labels[int(row_idx)].append(mapping[col_idx])
    return labels

def merge_preds(labels_1, labels_2, labels_2_real_instance_row_idx_set=None):
    if labels_2_real_instance_row_idx_set is None:
        for idx in labels_2.keys():
            labels_1[idx].extend(labels_2[idx])
    else:
        for idx, real_idx in enumerate(labels_2_real_instance_row_idx_set):
            labels_1[real_idx].extend(labels_2[idx])
    return labels_1

def _process_model(m_idx, model_info, tags_by_instance_l, labels_by_instance_l, tagset_files):
    # 1) Build feature/label matrices and gather timings
    (tagset_files_used, feature_matrix, label_matrix,
     instance_row_idx_set, instance_row_count, op_durations
    ) = tagsets_to_matrix(
        all_tags_l=model_info["all_tags_l"],
        tag_index_mapping=model_info["tag_index_mapping"],
        all_label_l=model_info["all_label_l"],
        label_index_mapping=model_info["label_index_mapping"],
        tags_by_instance_l=tags_by_instance_l,
        labels_by_instance_l=labels_by_instance_l,
        tagset_files=tagset_files,
        feature_importance=np.array(list(model_info["feature_importance"].values())),
        inference_flag=False
    )
    # 2) Run prediction if there is data
    if feature_matrix.size:
        t0 = time.time()
        preds = model_info["clf"].predict(feature_matrix)
        t1 = time.time()
        op_durations.update({
            "predict_time": t1 - t0,
            "feature_matrix_size": feature_matrix.size,
            "feature_matrix_xsize": feature_matrix.shape[0],
            "feature_matrix_ysize": feature_matrix.shape[1],
        })
        pred_label_name_d = one_hot_to_names("index_label_mapping", preds, mapping=model_info["mapping"])
        # Return durations and this model’s local predictions
        return m_idx, op_durations, (instance_row_idx_set, pred_label_name_d)
    else:
        # No features: return empty preds for all instances
        empty_pred = [(i, []) for i in range(len(tags_by_instance_l))]
        return m_idx, op_durations, ([], dict(empty_pred))

# -----------------------------
# Global Configuration & Model Loading
# -----------------------------
dataset = "data_4"
n_models = 1000  # adjust as needed
shuffle_idx = 0
test_sample_batch_idx = 0
n_samples = 4
clf_njobs = 32
n_estimators = 100
depth = 1
input_size = None
dim_compact_factor = 1
tree_method = "exact"
max_bin = 1
with_filter = True
freq = 25
cwd_clf = "/home/cc/ml-model-deployment/src_ss/models"

# Global list to hold loaded models.
models = []

def load_models():
    global models
    for i in range(n_models):
        clf_pathname = (
            f"{cwd_clf}/cwd_ML_with_{dataset}_{n_models}_{i}_train_"
            f"{shuffle_idx}shuffleidx_{test_sample_batch_idx}testsamplebatchidx_"
            f"{n_samples}nsamples_{clf_njobs}njobs_{n_estimators}trees_"
            f"{depth}depth_{input_size}-{dim_compact_factor}rawinput_sampling1_"
            f"{tree_method}treemethod_{max_bin}maxbin_modize_par_{with_filter}"
            f"{freq}removesharedornoisestags_verpak/model_init.json"
        )
        if not os.path.isfile(clf_pathname):
            raise Exception(f"Model file missing: {clf_pathname}")
        clf = xgb.XGBClassifier(
            max_depth=10, learning_rate=0.1, silent=False, objective='binary:logistic',
            booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1,
            max_delta_step=0, subsample=0.8, colsample_bytree=0.8,
            colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1
        )
        print(clf_pathname)
        clf.load_model(clf_pathname)
        mapping_path = clf_pathname.replace("model_init.json", "index_label_mapping")
        if not os.path.isfile(mapping_path):
            raise Exception(f"Mapping file missing for model {clf_pathname}: {mapping_path}")
        with open(mapping_path, 'rb') as fp:
            mapping = pickle.load(fp)
        feature_importance = {idx: imp for idx, imp in enumerate(clf.feature_importances_)}
        model_cwd = clf_pathname[:-15]
        # Load tag mappings if available.
        tag_mapping_path = model_cwd + "index_tag_mapping"
        tag_index_mapping_path = model_cwd + "tag_index_mapping"
        if os.path.isfile(tag_mapping_path) and os.path.isfile(tag_index_mapping_path):
            with open(tag_mapping_path, 'rb') as fp:
                all_tags_l = pickle.load(fp)
            with open(tag_index_mapping_path, 'rb') as fp:
                tag_index_mapping = pickle.load(fp)
        else:
            all_tags_l = None
            tag_index_mapping = None

        # Load label mappings.
        label_mapping_path = model_cwd + "index_label_mapping"
        label_index_mapping_path = model_cwd + "label_index_mapping"
        if os.path.isfile(label_mapping_path) and os.path.isfile(label_index_mapping_path):
            with open(label_mapping_path, 'rb') as fp:
                all_label_l = pickle.load(fp)
            with open(label_index_mapping_path, 'rb') as fp:
                label_index_mapping = pickle.load(fp)
        else:
            all_label_l = None
            label_index_mapping = None

        # Load iterative tag mappings if available.
        tag_mapping_iter_path = model_cwd + "index_tag_mapping_iter"
        tag_index_mapping_iter_path = model_cwd + "tag_index_mapping_iter"
        if os.path.isfile(tag_mapping_iter_path) and os.path.isfile(tag_index_mapping_iter_path):
            with open(tag_mapping_iter_path, 'rb') as fp:
                all_tags_l_iter = pickle.load(fp)
            with open(tag_index_mapping_iter_path, 'rb') as fp:
                tag_index_mapping_iter = pickle.load(fp)
        else:
            all_tags_l_iter = None
            tag_index_mapping_iter = None

        # Load iterative label mappings if available.
        label_mapping_iter_path = model_cwd + "index_label_mapping_iter"
        label_index_mapping_iter_path = model_cwd + "label_index_mapping_iter"
        if os.path.isfile(label_mapping_iter_path) and os.path.isfile(label_index_mapping_iter_path):
            with open(label_mapping_iter_path, 'rb') as fp:
                all_label_l_iter = pickle.load(fp)
            with open(label_index_mapping_iter_path, 'rb') as fp:
                label_index_mapping_iter = pickle.load(fp)
        else:
            all_label_l_iter = None
            label_index_mapping_iter = None

        models.append({
            "clf": clf,
            "mapping": mapping,  # For one_hot_to_names.
            "feature_importance": feature_importance,
            "cwd": model_cwd,
            "all_tags_l": all_tags_l,
            "tag_index_mapping": tag_index_mapping,
            "all_tags_l_iter": all_tags_l_iter,
            "tag_index_mapping_iter": tag_index_mapping_iter,
            "all_label_l": all_label_l,
            "label_index_mapping": label_index_mapping,
            "all_label_l_iter": all_label_l_iter,
            "label_index_mapping_iter": label_index_mapping_iter
        })

load_models()

# -----------------------------
# Flask Application & /predict Endpoint
# -----------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expected JSON format:
    {
      "samples": [
         { "instance_id": 0, "tags": ["tag1","tag2"], "true_labels": ["label1"] },
         …
      ]
    }
    Returns gzipped JSON:
      {
        "predictions": { … },
        "metrics": { … },
        "encoder_metrics": { … }
      }
    """
    # 1) Decompress incoming request if gzip-encoded
    if request.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            compressed = request.get_data()
            body = gzip.decompress(compressed)
            data = json.loads(body.decode("utf-8"))
        except Exception as e:
            return jsonify({"error": "Invalid gzip payload", "detail": str(e)}), 400
    else:
        data = request.get_json(force=True)

    samples = data.get('samples', [])
    if not samples:
        return jsonify({"error": "No samples provided"}), 400

    instance_ids = []
    tags_by_instance_l = []
    labels_by_instance_l = []
    tagset_files = []
    all_tags_set = set()
    all_label_set = set()
    for sample in samples:
        instance_ids.append(sample.get("instance_id"))
        raw_tags = sample.get("tags", [])
        instance_tags = {}
        for tag in raw_tags:
            # instance_tags[tag] = 1
            all_tags_set.add(tag)
        tags_by_instance_l.append(raw_tags)
        tagset_files.append(str(sample.get("instance_id")))
        true_labels = sample.get("true_labels", [])
        for label in true_labels:
            all_label_set.add(label)
        labels_by_instance_l.append(true_labels)


    # merged_predictions = defaultdict(list)
    # encoder_metrics = {}

    # # Process each model.
    # for m_idx, model_info in enumerate(models):
    #     tagset_files_used, feature_matrix, label_matrix, instance_row_idx_set, instance_row_count, op_durations = tagsets_to_matrix(
    #         all_tags_l=model_info["all_tags_l"],
    #         tag_index_mapping=model_info["tag_index_mapping"],
    #         all_label_l=model_info["all_label_l"],
    #         label_index_mapping=model_info["label_index_mapping"],
    #         tags_by_instance_l=tags_by_instance_l,
    #         labels_by_instance_l=labels_by_instance_l,
    #         tagset_files=tagset_files,
    #         feature_importance=np.array(list(model_info["feature_importance"].values())),
    #         inference_flag=False
    #     )
    #     # encoder_metrics[f"model_{m_idx}"] = op_durations
        
    #     if feature_matrix.size != 0:
    #         t_predict_0 = time.time()
    #         pred_label_matrix = model_info["clf"].predict(feature_matrix)
    #         t_predict_t = time.time()
    #         op_durations["predict_time"] = t_predict_t - t_predict_0
    #         op_durations["feature_matrix_size"] = feature_matrix.size
    #         op_durations["feature_matrix_xsize"] = feature_matrix.shape[0]
    #         op_durations["feature_matrix_ysize"] = feature_matrix.shape[1]
    #         pred_label_name_d = one_hot_to_names("index_label_mapping", pred_label_matrix, mapping=model_info["mapping"])
    #         merged_predictions = merge_preds(merged_predictions, pred_label_name_d, instance_row_idx_set)
    #     else:
    #         for idx in range(len(instance_ids)):
    #             merged_predictions.setdefault(idx, [])
    #     encoder_metrics[f"model_{m_idx}"] = op_durations

    # predictions = {inst_id: merged_predictions.get(idx, []) for idx, inst_id in enumerate(instance_ids)}




    merged_predictions = defaultdict(list)
    encoder_metrics = {}

    # 3) Parallel dispatch
    with ProcessPoolExecutor(max_workers=min(len(models), os.cpu_count() or 1)) as executor:
        futures = [
            executor.submit(
                _process_model, m_idx, mi,
                tags_by_instance_l, labels_by_instance_l, tagset_files
            )
            for m_idx, mi in enumerate(models)
        ]
        for future in as_completed(futures):
            m_idx, op_durations, (instance_row_idx_set, model_preds) = future.result()
            encoder_metrics[f"model_{m_idx}"] = op_durations
            # 4) Merge this model’s preds into the global map
            merged_predictions = merge_preds(merged_predictions, model_preds, instance_row_idx_set)
            # print("Merged predictions:", merged_predictions)
    
    # 5) Finalize JSON response as before
    predictions = {
        inst_id: merged_predictions.get(idx, [])
        for idx, inst_id in enumerate(instance_ids)
    }

    metrics = {}
    if any(len(lbls) > 0 for lbls in labels_by_instance_l):
        all_labels = sorted(
            set(label for labels in labels_by_instance_l for label in labels) |
            set(label for labels in predictions.values() for label in labels)
        )
        if all_labels:
            mlb = MultiLabelBinarizer(classes=all_labels)
            true_binarized = mlb.fit_transform(labels_by_instance_l)
            predicted_ordered = [predictions.get(inst_id, []) for inst_id in instance_ids]
            predicted_binarized = mlb.transform(predicted_ordered)
            metrics['accuracy'] = accuracy_score(true_binarized, predicted_binarized)
            
            metrics['f1_score_weighted'] = f1_score(true_binarized, predicted_binarized, average='weighted')
            metrics['f1_score_macro'] = f1_score(true_binarized, predicted_binarized, average='macro')
            metrics['f1_score_micro'] = f1_score(true_binarized, predicted_binarized, average='micro')
            
            metrics['precision_weighted'] = precision_score(true_binarized, predicted_binarized, average='weighted')
            metrics['precision_macro'] = precision_score(true_binarized, predicted_binarized, average='macro')
            metrics['precision_micro'] = precision_score(true_binarized, predicted_binarized, average='micro')
            
            metrics['recall_weighted'] = recall_score(true_binarized, predicted_binarized, average='weighted')
            metrics['recall_macro'] = recall_score(true_binarized, predicted_binarized, average='macro')
            metrics['recall_micro'] = recall_score(true_binarized, predicted_binarized, average='micro')
            
            metrics['hamming_loss'] = hamming_loss(true_binarized, predicted_binarized)

    response = {
        "predictions": predictions,
        "metrics": metrics,
        "encoder_metrics": encoder_metrics
    }

    # 2) Serialize and gzip-compress the JSON response
    resp_bytes = json.dumps(response).encode("utf-8")
    gzipped = gzip.compress(resp_bytes)

    return Response(
        gzipped,
        status=200,
        mimetype="application/json",
        headers={"Content-Encoding": "gzip"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
