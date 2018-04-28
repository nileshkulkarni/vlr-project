import numpy as np
import os
import collections
from pandas import DataFrame
from baseline.utils_1 import read_class_names


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU



def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.1, 0.9, 9)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = np.zeros(len(tiou_thresholds))
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

    return ap




def find_average_precision(detections_cls, ground_truths_cls, tiou_thresholds):
    #import pdb;pdb.set_trace()
    ap_class = compute_average_precision_detection(ground_truths_cls, detections_cls, tiou_thresholds=tiou_thresholds)


    return ap_class




def read_detections( class_names, data_directory):
    test_anno_dir = data_directory
    detections_all = []
    #import pdb;pdb.set_trace()
    for key in class_names:
        detections_class = []#collections.defaultdict(list)
        class_name = class_names[key]
        ipfilename = os.path.join(test_anno_dir, class_name)
        ipfilename = ipfilename + '_detections.csv'
        detection_file = open(ipfilename , 'r')
        lines = detection_file.readlines()
        for j, line in enumerate(lines):
            line = line.strip().split(',')
            detection = []
            video_name = line[0] #video_name
            detection.append(video_name)
            detection.append(float(line[1])) #t_begin
            detection.append(float(line[2])) #t_end
            detection.append(float(line[3])) #score
            detections_class.append(detection)

        detections_all.append(detections_class)

    dets_dfs = []
    for i in range(len(detections_all)):
        df = DataFrame(detections_all[i])
        df.columns = ['video-id', 't-start', 't-end', 'score']
        dets_dfs.append(df)


    return dets_dfs


def read_groundtruths(class_names, data_directory = '/scratch/smynepal/THUMOSFrames/temporal_annotations/'):

    #reading each class file

    test_anno_dir = os.path.join(data_directory, 'test/annotations/')
    ground_truths_all = []
    for key in class_names:
        class_name = class_names[key]
        anno_filename = class_name + '_test.txt'
        anno_filename = os.path.join(test_anno_dir, anno_filename)
        anno_file = open(anno_filename, 'r')
        lines = anno_file.readlines()
        gts_class = []#collections.defaultdict(list)
        for i, line in enumerate(lines):
            line = line.strip().split()
            anno = []
            video_name = line[0]  # video_name
            anno.append(video_name)
            anno.append(float(line[1]))  # t_begin
            anno.append(float(line[2]))  # t_end
            gts_class.append(anno)

        ground_truths_all.append(gts_class)
    gt_dfs = []
    for i in range(len(ground_truths_all)):
        df = DataFrame(ground_truths_all[i])
        #df = df.transpose()
        df.columns = ['video-id', 't-start', 't-end']
        gt_dfs.append(df)
    return gt_dfs



data_directory = '/scratch/smynepal/THUMOSFrames/temporal_annotations/'
class_names = read_class_names(data_directory, class_names_file='class_dict.csv')

gts = read_groundtruths(class_names, data_directory=data_directory)
preds = read_detections(class_names, data_directory='/home/smynepal/Projects/VLR/I3D/train/vlr-project/weakly-supvervized-temp/predictions/threshold_tcam_01_strat3_one_layer')

tiou_thresholds=np.linspace(0.1, 0.9, 9)
numClasses = 20
ap = np.zeros((len(tiou_thresholds), numClasses))

for i in range(numClasses):
    print('processing :', class_names[i])
    ap[:, i] = find_average_precision(preds[i], gts[i], tiou_thresholds)

aap = np.mean(ap, axis = 1)
for i in range(ap.shape[0]):
    print('rgb AP at IoU :', tiou_thresholds[i], 'is ', aap[i])
#print(ap[0,:])
import pdb;pdb.set_trace()
np.save('APs_01_09_9_threshold_015_strat3.npy', ap)