"""Evaluate directional marking point detector."""
import json
import os
import cv2 as cv
import numpy as np
import torch
import config
import util
from util import *
from data import match_slots, Slot
from model import DirectionalPointDetector
from inference import detect_marking_points, inference_slots
from model_slot_det import get_model
import tqdm
from inference import plot_points
from PIL import Image
from losses_0807 import calc_precision_recall,collect_error,compute_eval_results,collect_error
from process_0807 import MarkingPoint


def get_ground_truths(label):
    """Read label to get ground truth slot."""
    slots = np.array(label['slots'])
    if slots.size == 0:
        return []
    if len(slots.shape) < 2:
        slots = np.expand_dims(slots, axis=0)
    marks = np.array(label['marks'])
    if len(marks.shape) < 2:
        marks = np.expand_dims(marks, axis=0)
    ground_truths = []
    for slot in slots:
        mark_a = marks[slot[0] - 1]
        mark_b = marks[slot[1] - 1]
        coords = np.array([mark_a[0], mark_a[1], mark_b[0], mark_b[1]])
        coords = (coords - 0.5) / 600
        ground_truths.append(Slot(*coords))
    return ground_truths


def psevaluate_train(args,model,val_loader,device):
    """Evaluate directional point detector."""

    dp_detector = model.eval()

    
    ground_truths_list = []
    predictions_list = []
    no_slot_count = 0
    slot_count = 0
    batch_num = args.batch_size
    # for idx, (images, marking_points) in enumerate(val_loader):
    for idx, (image,labels) in enumerate(val_loader):
    # for idx, label_file in enumerate(os.listdir(args.val_labels_directory)):
        # if ".jpg" in label_file:
        #     continue
        # if idx > 200:
        #     break
        # print(idx)
        images = torch.stack(image).to(device)

        pred_points = detect_marking_points(
            dp_detector, images, config.CONFID_THRESH_FOR_POINT, device,isval=True)

        
        for idx in range(len(pred_points)):
            pred_slots = []
            for slot in pred_points[idx]:
                # point_ax = slot[1].x
                # point_ay = slot[1].y
                # point_bx = slot[1].x + slot[1].lenEntryLine_x
                # point_by = slot[1].y + slot[1].lenEntryLine_y
                prob = slot[0]
                pred_slots.append(
                    (prob, MarkingPoint(slot[1].x, slot[1].y, slot[1].lenSepLine_x, \
                                        slot[1].lenSepLine_y,slot[1].lenEntryLine_x,slot[1].lenEntryLine_y)))
            predictions_list.append(pred_slots)

        for idx in range(len(labels)):
            ground_truths_list.append(labels[idx])

        # img2show1 = Image.fromarray(tensor2array(image[0]))
        # img2show2 = Image.fromarray(tensor2array(image[1]))
        # img2show3 = Image.fromarray(tensor2array(image[2]))
        # img2show4 = Image.fromarray(tensor2array(image[3]))
        # img2show1.show()
        # img2show2.show()
        # img2show3.show()
        # img2show4.show()
        #
        # ccc = 1
        # slotimg1 = plot_points(tensor2array(image[0]), pred_points[0])
        # slotimg1.show()
        # slotimg2 = plot_points(tensor2array(image[1]), pred_points[1])
        # slotimg2.show()
        # slotimg3 = plot_points(tensor2array(image[2]), pred_points[2])
        # slotimg3.show()
        # slotimg4 = plot_points(tensor2array(image[3]), pred_points[3])
        # slotimg4.show()
        

    
        # if idx > 500:
        #     break
    

    
    precisions, recalls, TP, FP, TN, FN, Precision, Recall, Accuracy \
        = calc_precision_recall(ground_truths_list, predictions_list)

    # err_result = collect_error(ground_truths_list, predictions_list)


    
    print("TP: {} || FP: {} || TN: {} || FN: {} || Precision: {} || Recall: {} || Accuracy: {}" \
          .format(TP,FP,TN,FN,Precision,Recall,Accuracy))
    

    return Precision
    # if args.enable_visdom:
    #     logger.plot_curve(precisions, recalls)
    # logger.log(average_precision=average_precision)
    
    

def psevaluate_detector(args):
    """Evaluate directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)

    # dp_detector = DirectionalPointDetector(
    #     3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    
    dp_detector = get_model().to(device)

    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights,map_location='cuda:0'))
    dp_detector.eval()

    logger = util.Logger(enable_visdom=args.enable_visdom)

    ground_truths_list = []
    predictions_list = []
    no_slot_count = 0
    slot_count = 0
    for idx, label_file in enumerate(os.listdir(args.label_directory)):
        # if ".jpg" in label_file:
        #     continue
        # if idx > 200:
        #     break
        label_path = os.path.join(args.label_directory,label_file)
        with open(label_path, 'r') as file:
            jsonlabel = json.load(file)

        # slots = jsonlabel["slots"]

        centralied_marks = np.array(jsonlabel['marks'])
        slots = np.array(jsonlabel['slots'])
        if len(slots) == 0:
            no_slot_count += 1
            continue
        else:
            slot_count += 1
            

        if len(centralied_marks.shape) < 2:
            centralied_marks = np.expand_dims(centralied_marks, axis=0)
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)
            
            
        name = os.path.splitext(label_file)[0]
        # print(idx, name)
        read_path = os.path.join(args.image_directory, name + '.jpg')
        print(read_path)
        image = cv.imread(read_path)
        pred_points = detect_marking_points(
            dp_detector, image, config.CONFID_THRESH_FOR_POINT, device)
        slots = []

        slotimg = plot_points(image, pred_points[0])
        save_directory = "/media/fjy/SHARE/dataset/ps2.0/test_result_show"
        save_path = os.path.join(save_directory, name + '.jpg')
        print(save_path)
        slotimg.save(save_path)
        
        # marking_points = pred_points
        pred_slots = []
        for slot in pred_points[0]:
            point_ax = slot[1].x
            point_ay = slot[1].y
            point_bx = slot[1].x + slot[1].dx
            point_by = slot[1].y + slot[1].dy
            prob = slot[0]
            pred_slots.append(
                (prob, Slot(point_ax, point_ay, point_bx, point_by)))
        predictions_list.append(pred_slots)
            
            
        # if pred_points:
        #     marking_points = list(list(zip(*pred_points))[1])
        #     slots = inference_slots(marking_points)
        # pred_slots = []
        # for slot in slots:
        #     point_b = marking_points[slot[1]]
        #     prob = min((pred_points[slot[0]][0], pred_points[slot[1]][0]))
        #     pred_slots.append(
        #         (prob, Slot(point_a.x, point_a.y, point_b.x, point_b.y)))
        # predictions_list.append(pred_slots)

        with open(os.path.join(args.label_directory, label_file), 'r') as file:
            ground_truths_list.append(get_ground_truths(json.load(file)))
            
        # if idx > 500:
        #     break

    precisions, recalls = util.calc_precision_recall(
        ground_truths_list, predictions_list, match_slots)
    average_precision = util.calc_average_precision(precisions, recalls)
    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)
    logger.log(average_precision=average_precision)


if __name__ == '__main__':
    psevaluate_detector(config.get_parser_for_ps_evaluation().parse_args())
