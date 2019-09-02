from itertools import count

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.
    
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    assert len(bbox1) == 4
    assert len(bbox2) == 4
    s1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    intersection_rows = max(min((bbox2[2], bbox1[2])) - max((bbox2[0], bbox1[0])), 0)
    intersection_cols = max(min((bbox2[3], bbox1[3])) - max((bbox2[1], bbox1[1])), 0)
    intersection = intersection_rows * intersection_cols
    union = s1 + s2 - intersection
    return 1.0 * intersection / union

def best_match(pred_bboxes, true_bboxes, decision_function):
    """Calculate best matching between two bbox lists.
    """
    matched, false_negative, false_positive = [], [], []
    
    _pred_bboxes = np.array(pred_bboxes, dtype=np.int32) # Make a copy
    _true_bboxes = np.array(true_bboxes, dtype=np.int32)
    _decision_function = np.array(decision_function, dtype=np.int32)
    image_indeces = sorted(set(_pred_bboxes[:, 0]).union(set(_true_bboxes[:, 0])))
    
    for image_index in image_indeces:
        
        pred_indeces = np.where(_pred_bboxes[:, 0] == image_index)[0]
        order = np.argsort(_decision_function[pred_indeces])[::-1]
        pred_indeces = pred_indeces[order]
        decision_function = _decision_function[pred_indeces]
        pred_bboxes = _pred_bboxes[pred_indeces]
        
        true_indeces = np.where(_true_bboxes[:, 0] == image_index)[0]
        # true_bboxes = _true_bboxes[true_indeces]
        matched_true_bboxes = set()

        for j, pred_bbox in enumerate(pred_bboxes):
            best_match = -1
            best_score = 0.1

            for i in true_indeces:
                true_bbox = _true_bboxes[i]
                match_score = iou_score(true_bbox[1:5], pred_bbox[1:5])
                if (i not in matched_true_bboxes) and (match_score > best_score):
                    best_match = i
                    best_score = match_score

            if best_match != -1:
                matched.append((best_match, pred_indeces[j]))
                matched_true_bboxes.add(best_match)

            else:
                false_positive.append(pred_indeces[j])

        for i in true_indeces:
            if i not in matched_true_bboxes:
                false_negative.append(i)
     
    return matched, false_negative, false_positive

def average_precision(precision, recall):
    """Calculate score between two sets of bboxes."""
    recall = np.array(recall) * 10
    
    result = precision[0]

    step = 0
    for i in range(1, len(precision)):
        prev_x, next_x = recall[i - 1], recall[i]
        prev_y, next_y = precision[i - 1], precision[i]
        
        while step + 1 <= next_x + 1e-6:
            step += 1
            result += ((next_x - step) * prev_y + (step - prev_x) * next_y) / (next_x - prev_x)
   
    return result / 11

def auc(x, y):
    """Calculate auc score."""
    x, y = np.array(x), np.array(y)
    order = x.argsort()
    x = x[order]
    y = y[order]
    
    result = 0

    for i in range(1, len(x)):
        prev_x, next_x = x[i - 1], x[i]
        prev_y, next_y = y[i - 1], y[i]
        
        result += (next_x - prev_x) * (prev_y + next_y) / 2.0
   
    return result