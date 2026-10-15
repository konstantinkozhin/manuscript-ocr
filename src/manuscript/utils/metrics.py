from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from .geometry import _box_iou


def _match_boxes(
    pred_boxes: List[Tuple[float, float, float, float]],
    gt_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Match predicted boxes to ground truth boxes using IoU threshold.
    
    Uses greedy matching: pairs with highest IoU are matched first.
    
    Parameters
    ----------
    pred_boxes : list of tuple
        Predicted boxes in format (x_min, y_min, x_max, y_max).
    gt_boxes : list of tuple
        Ground truth boxes in same format.
    iou_threshold : float, default=0.5
        Minimum IoU to consider a match.
        
    Returns
    -------
    tp : int
        True positives (matched predictions).
    fp : int
        False positives (unmatched predictions).
    fn : int
        False negatives (unmatched ground truths).
        
    Examples
    --------
    >>> pred = [(10, 10, 50, 50), (60, 60, 100, 100)]
    >>> gt = [(12, 12, 52, 52)]  # Overlaps with first prediction
    >>> tp, fp, fn = _match_boxes(pred, gt, iou_threshold=0.5)
    >>> (tp, fp, fn)
    (1, 1, 0)
    """
    # Handle edge cases
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = _box_iou(pred, gt)
    
    # Greedy matching: sort all (iou, pred_idx, gt_idx) tuples by IoU
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
    
    matches.sort(reverse=True)  # Highest IoU first
    
    # Track which predictions and GTs have been matched
    matched_pred = set()
    matched_gt = set()
    
    for iou_val, i, j in matches:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
    
    # Calculate metrics
    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn

def _compute_f1_score(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Tuple[float, float, float]:
    """
    Compute F1 score, precision, and recall from TP/FP/FN counts.
    
    Parameters
    ----------
    true_positives : int
        Number of true positives.
    false_positives : int
        Number of false positives.
    false_negatives : int
        Number of false negatives.
        
    Returns
    -------
    f1 : float
        F1 score in range [0, 1].
    precision : float
        Precision in range [0, 1].
    recall : float
        Recall in range [0, 1].
        
    Examples
    --------
    >>> f1, prec, rec = _compute_f1_score(80, 10, 10)
    >>> round(f1, 2)
    0.84
    >>> round(prec, 2)
    0.89
    >>> round(rec, 2)
    0.89
    
    >>> # Perfect detection
    >>> _compute_f1_score(100, 0, 0)
    (1.0, 1.0, 1.0)
    
    >>> # No detections
    >>> _compute_f1_score(0, 0, 10)
    (0.0, 0.0, 0.0)
    """
    if true_positives == 0:
        return 0.0, 0.0, 0.0
    
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    
    if precision + recall == 0:
        return 0.0, precision, recall
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def _evaluate_image_worker(args):
    """Worker function for parallel evaluation (used by evaluate_dataset)."""
    image_id, pred_boxes, gt_boxes, iou_thresholds = args
    results = {}
    
    for threshold in iou_thresholds:
        tp, fp, fn = _match_boxes(pred_boxes, gt_boxes, iou_threshold=threshold)
        results[threshold] = (tp, fp, fn)
    
    return results

def _evaluate_dataset(
    predictions: Dict[str, List[Tuple[float, float, float, float]]],
    ground_truths: Dict[str, List[Tuple[float, float, float, float]]],
    iou_thresholds: Optional[List[float]] = None,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate object detection on a dataset with multiple IoU thresholds.
    
    Computes precision, recall, and F1 at various IoU thresholds, similar
    to COCO-style evaluation.
    
    Parameters
    ----------
    predictions : dict
        Dictionary mapping image IDs to lists of predicted boxes.
        Each box: (x_min, y_min, x_max, y_max).
    ground_truths : dict
        Dictionary mapping image IDs to lists of ground truth boxes.
    iou_thresholds : list of float, optional
        IoU thresholds to evaluate at. Default: [0.5, 0.55, ..., 0.95].
    verbose : bool, default=True
        If True, show progress bar.
    n_jobs : int, optional
        Number of parallel workers. Default: use all CPUs.
        Set to 1 to disable parallelization.
        
    Returns
    -------
    dict
        Dictionary with metrics:
        - 'f1@0.5', 'precision@0.5', 'recall@0.5': Metrics at IoU=0.5
        - 'f1@0.5:0.95': Mean F1 across all thresholds (COCO-style)
        - 'tp@X', 'fp@X', 'fn@X': Raw counts for each threshold
        - 'num_images': Total images evaluated
        - 'num_predictions': Total prediction boxes
        - 'num_ground_truths': Total ground truth boxes
        
    Examples
    --------
    >>> predictions = {
    ...     'img1': [(10, 10, 50, 50), (60, 60, 100, 100)],
    ...     'img2': [(20, 20, 80, 80)]
    ... }
    >>> ground_truths = {
    ...     'img1': [(12, 12, 52, 52)],
    ...     'img2': [(22, 22, 82, 82)]
    ... }
    >>> results = _evaluate_dataset(predictions, ground_truths, verbose=False)
    >>> results['f1@0.5'] > 0.5
    True
    
    Notes
    -----
    - Uses parallel processing by default for faster evaluation
    - Missing images in predictions/ground_truths are treated as empty
    - Results comparable to standard object detection benchmarks
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    
    # Initialize accumulators
    total_tp = {th: 0 for th in iou_thresholds}
    total_fp = {th: 0 for th in iou_thresholds}
    total_fn = {th: 0 for th in iou_thresholds}
    
    # Get all image IDs
    all_image_ids = list(set(list(predictions.keys()) + list(ground_truths.keys())))
    
    # Determine if we should use parallel processing
    use_parallel = n_jobs is None or n_jobs > 1
    
    if use_parallel:
        import multiprocessing as mp
        from multiprocessing import Pool
        
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        
        # Prepare arguments for workers
        worker_args = [
            (
                image_id,
                predictions.get(image_id, []),
                ground_truths.get(image_id, []),
                iou_thresholds,
            )
            for image_id in all_image_ids
        ]
        
        # Use spawn context for compatibility
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            if verbose:
                image_results = list(
                    tqdm(
                        pool.imap(_evaluate_image_worker, worker_args),
                        total=len(worker_args),
                        desc="Evaluating images",
                    )
                )
            else:
                image_results = pool.map(_evaluate_image_worker, worker_args)
        
        # Aggregate results
        for result in image_results:
            for threshold, (tp, fp, fn) in result.items():
                total_tp[threshold] += tp
                total_fp[threshold] += fp
                total_fn[threshold] += fn
    else:
        # Sequential evaluation
        iterator = (
            tqdm(all_image_ids, desc="Evaluating images") if verbose else all_image_ids
        )
        
        for image_id in iterator:
            pred_boxes = predictions.get(image_id, [])
            gt_boxes = ground_truths.get(image_id, [])
            
            for threshold in iou_thresholds:
                tp, fp, fn = _match_boxes(pred_boxes, gt_boxes, iou_threshold=threshold)
                total_tp[threshold] += tp
                total_fp[threshold] += fp
                total_fn[threshold] += fn
    
    # Compute metrics for each threshold
    results = {}
    f1_scores = []
    
    for threshold in iou_thresholds:
        tp = total_tp[threshold]
        fp = total_fp[threshold]
        fn = total_fn[threshold]
        
        f1, precision, recall = _compute_f1_score(tp, fp, fn)
        
        # Store with formatted keys
        results[f"f1@{threshold:.2f}"] = f1
        results[f"precision@{threshold:.2f}"] = precision
        results[f"recall@{threshold:.2f}"] = recall
        results[f"tp@{threshold:.2f}"] = tp
        results[f"fp@{threshold:.2f}"] = fp
        results[f"fn@{threshold:.2f}"] = fn
        
        f1_scores.append(f1)
    
    # Add summary metrics
    results["f1@0.5"] = results.get("f1@0.50", 0.0)
    results["precision@0.5"] = results.get("precision@0.50", 0.0)
    results["recall@0.5"] = results.get("recall@0.50", 0.0)
    results["f1@0.5:0.95"] = float(np.mean(f1_scores))
    results["num_images"] = len(all_image_ids)
    results["num_predictions"] = sum(len(boxes) for boxes in predictions.values())
    results["num_ground_truths"] = sum(len(boxes) for boxes in ground_truths.values())
    
    return results
