import torch
import torch.nn
import os.path as osp
from baseline.utils.parser import get_opts
from baseline.nnutils.stream_modules import ActionClassification
from baseline.data.ucf101 import UCF101, split
from baseline.logger import Logger
from baseline.utils_1 import non_maximal_suppression as nms
from baseline.utils_1 import read_class_names
import pdb
import collections
from sklearn.metrics import average_precision_score, recall_score
from scipy.ndimage import label
import os

import numpy as np


def process_tcam(attn, tcam, numInputs):
    # assuming input tcam is 400 x 20
    # 400 segments
    # assuming center of value in tcam designates activation at center of segment
    if numInputs < 400:
        return attn[:numInputs], tcam[:numInputs]
    else:
        
        sample_range = numInputs
        seg_length = 400
        sampledN = np.round(np.linspace(0, sample_range, seg_length + 1)).astype(np.int32)
        K = sampledN[0:-1]
        K = np.mod(K, np.ones(K.shape) * numInputs).astype(np.int)
        probe_points = [x for x in range(numInputs)]
        samples = K
        #import pdb;pdb.set_trace()
        interpolated_tcam = np.zeros([len(probe_points), tcam.shape[1]])
        for j in range(tcam.shape[1]):
            interpolated_tcam[:, j] = np.interp(probe_points, samples, tcam[:, j])
        interpolated_attn = np.interp(probe_points, samples, attn[:,0])
        interpolated_attn = np.expand_dims(interpolated_attn, axis=1)
        return interpolated_attn, interpolated_tcam


def threshold_tcams(tcam, threshold=0.3):
    # assuming input is timeframes x 20
    segmented_tcam = (tcam > threshold) * 1
    return segmented_tcam


def find_proposals(seg_tcam_flow, seg_tcam_rgb):
    proposals = []
    for i in range(seg_tcam_flow.shape[1]):  # num classes
        seg_tcam_class = seg_tcam_flow[:, i]
        proposals_map, num_proposals = label(seg_tcam_class)
        proposals_class = []
        for j in range(1, num_proposals + 1):
            tmin = np.min(np.where(proposals_map == j)[0])
            tmax = np.max(np.where(proposals_map == j)[0])
            proposal = tuple([tmin, tmax])
            proposals_class.append(proposal)

        seg_tcam_class = seg_tcam_rgb[:, i]
        proposals_map, num_proposals = label(seg_tcam_class)

        # num_proposals = np.max(proposals_map) + 1
        for j in range(1, num_proposals + 1):
            tmin = np.min(np.where(proposals_map == j)[0])
            tmax = np.max(np.where(proposals_map == j)[0])
            proposal = tuple([tmin, tmax])
            proposals_class.append(proposal)
        proposals.append(proposals_class)

    return proposals


def find_scores(proposals, tcam_flow, tcam_rgb, attn_flow, attn_rgb, flow_mode_weight=0.5):
    proposal_scores = []
    detections = []
    for cls_idx in range(len(proposals)):  # num classes
        proposals_class = proposals[cls_idx]
        detections_class = []
        for i in range(len(proposals_class)):
            t_begin = proposals_class[i][0]
            t_end = proposals_class[i][1]
            proposal_score = 0
            if t_end != t_begin:
                for i in range(t_begin, t_end):
                    proposal_score += flow_mode_weight * tcam_flow[i, cls_idx] * attn_flow[i] + (
                                                                                                    1 - flow_mode_weight) * \
                                                                                                tcam_rgb[i, cls_idx] * \
                                                                                                attn_rgb[i]
            else:
                proposal_score = flow_mode_weight * tcam_flow[t_begin, cls_idx] * attn_flow[t_begin] + (
                                                                                                           1 - flow_mode_weight) * \
                                                                                                       tcam_rgb[
                                                                                                           t_begin, cls_idx] * \
                                                                                                       attn_rgb[t_begin]

            proposal_score = proposal_score / (t_end - t_begin + 1)
            detections_class.append(tuple([t_begin, t_end, proposal_score[0]]))
        detections.append(detections_class)

    return detections


def infer(model, data_iter, fps=16):
    # assuming input data to be similar to the input received by model during training
    model.eval()
    fin_detections = collections.defaultdict(list)
    class_probs = collections.defaultdict(list)

    for i, batch in enumerate(data_iter):
        input_data = batch
        numInputs = input_data['numInputs'].data.numpy()[0][0][0]

        
        #import pdb;pdb.set_trace()
        outputs = model(input_data['rgb'], input_data['flow'])
        tcam_flow = np.squeeze(outputs['tcam_flow'].data.cpu().numpy(), axis=0)
        tcam_rgb = np.squeeze(outputs['tcam_rgb'].data.cpu().numpy(), axis=0)
        attn_rgb = np.squeeze(outputs['attn_rgb'].data.cpu().numpy(), axis=0)
        attn_flow = np.squeeze(outputs['attn_flow'].data.cpu().numpy(), axis=0)
        attn_rgb, tcam_rgb = process_tcam(attn_rgb, tcam_rgb, numInputs)
        attn_flow, tcam_flow = process_tcam(attn_flow, tcam_flow, numInputs)


        attn_rgb = attn_rgb * 0 
        weighted_tcam_flow = attn_flow * (1.0 / (1 + np.exp(-tcam_flow)))
        weighted_tcam_rgb = attn_rgb * (1.0 / (1 + np.exp(-tcam_rgb)))
        #import pdb;pdb.set_trace()
        seg_tcam_flow = threshold_tcams(weighted_tcam_flow, threshold = 0.1)
        seg_tcam_rgb = threshold_tcams(weighted_tcam_rgb, threshold = 0.1)


        #classes
        class_rgb = np.squeeze(outputs['class_rgb'].data.cpu().numpy(), axis=0)
        class_flow = np.squeeze(outputs['class_flow'].data.cpu().numpy(), axis=0)
        class_both = class_rgb + class_flow
        probs = 1.0/(1 + np.exp(-class_both))
        proposals = find_proposals(seg_tcam_flow, seg_tcam_rgb)

        detections = find_scores(proposals, tcam_flow, tcam_rgb, attn_flow, attn_rgb)
        # How is NMS defined for temporal islands?
        final_proposals = nms(detections, threshold=0.2)
        fin_detections[i] = final_proposals
        #if i==4:import pdb;pdb.set_trace()
        class_probs[i] = probs
    return fin_detections, class_probs


def collate_fn(batch):
    collated_batch = dict()
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = torch.stack([example[key] for example in batch])
    return collated_batch


def write_detections_gt_style(fin_detections, class_probs, video_names, save_dir, class_names, fps=10):
    fps = float(fps)
    numClasses = len(fin_detections[0])
    for j in range(numClasses):
        class_name = class_names[j]
        opfile = open(os.path.join(save_dir, class_name + '_detections.csv'), 'w')
        for i in range(len(fin_detections)):
            if not fin_detections[i][j]:continue
            cls_detections = fin_detections[i][j]
            cls_prob = class_probs[i][j]
            if cls_prob >  0.1:
                for cls_detection in cls_detections:

                    t_begin = cls_detection[0]
                    t_end = cls_detection[1]
                    score = cls_detection[2]
                    #if score < 0:continue
                    t_begin = 16 * t_begin / fps  # time in seconds, #fps
                    t_end = 16 * float(t_end) / float(fps) + 15.0 / float(fps)  # beginning of one segment to end of next
                    opfile.write(video_names[i] + ',' + str(t_begin) + ',' + str(t_end) + ',' + str(score) + '\n')
            else:
                continue

def read_video_order_file(mode='test'):
    ipfilename = './baseline/labels/video_indices_' + mode + '.csv'
    ipfile = open(ipfilename)
    lines = ipfile.readlines()
    filenames = []
    for i, line in enumerate(lines):
        filenames.append(line.strip().split(',')[0])
    return filenames


from time import gmtime, strftime


def main(opts):
    class_names = read_class_names(data_directory='/scratch/smynepal/THUMOSFrames/temporal_annotations/',
                                   class_names_file='class_dict.csv')
    dataset_test = UCF101('test', opts)

    data_iter_test = torch.utils.data.DataLoader(dataset_test,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 collate_fn=collate_fn)

    logdir = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    opts.logger = logger = Logger(osp.join(opts.cache_dir, 'logs', logdir),
                                  'baseline', )
    fps = opts.fps
    action_net = ActionClassification(opts.feature_size, opts.num_classes, opts)
    action_net_trained = torch.load(
        '/home/smynepal/Projects/VLR/I3D/train/vlr-project/weakly-supvervized-temp/saved_models/action_net_reg_00001_one_layer.pkl')
    action_net.load_state_dict(action_net_trained['state_dict'])
    action_net.cuda()
    fin_detections, class_probs = infer(action_net, data_iter_test, fps)
    #import pdb;pdb.set_trace()
    video_names = read_video_order_file(mode='test')
    write_detections_gt_style(fin_detections, class_probs, video_names,
                              save_dir='/home/smynepal/Projects/VLR/I3D/train/vlr-project/weakly-supvervized-temp/predictions/threshold_tcam_01_strat3_one_layer_flow/',
                              class_names=class_names)

    return



if __name__ == "__main__":
    print('here')
    opts = get_opts()
    main(opts)
