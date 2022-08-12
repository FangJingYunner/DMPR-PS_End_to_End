"""Inference demo of directional point detector."""
import math
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import pair_marking_points, calc_point_squre_dist, pass_through_third_point,get_predicted_points_dmpr
from model import DirectionalPointDetector
from util import Timer
from PIL import Image, ImageDraw, ImageFont
from model_slot_det import get_model
from process_0807 import get_predicted_points
def plot_points(image, pred_points):
    """Plot marking points on the image."""

    height = image.shape[0]
    width = image.shape[1]

    img2show = Image.fromarray(image)
    imgdraw = ImageDraw.Draw(img2show)
    if not pred_points:
        return img2show
    
    color_bias = 0
    for confidence, marking_point in pred_points:
        pa_x = width * marking_point.x - 0.5
        pa_y = height * marking_point.y - 0.5
        pb_x = width * (marking_point.x + marking_point.dx) - 0.5
        pb_y = height * (marking_point.y + marking_point.dy) - 0.5

        cos_val = math.cos(marking_point.direction)
        sin_val = math.sin(marking_point.direction)
        pd_x = pa_x + 50*cos_val
        pd_y = pa_y + 50*sin_val
        
        pc_x = pb_x + 50*cos_val
        pc_y = pb_y + 50*sin_val
        
        # cv.line(image, (pa_x, pa_y), (pb_x, pb_y), (0, 0, 255), 2)


        imgdraw.line((pa_x, pa_y, pb_x,pb_y), fill=(80+color_bias,120+color_bias,120), width=8)
        imgdraw.line((pa_x, pa_y, pd_x, pd_y), fill='red', width=8)
        imgdraw.line((pc_x, pc_y, pb_x, pb_y), fill='yellow', width=8)

        color_bias = color_bias + 50
    # img2show.show()
    return img2show
        # cos_val = math.cos(marking_point.direction)
        # sin_val = math.sin(marking_point.direction)
        # p1_x = p0_x + 50*cos_val
        # p1_y = p0_y + 50*sin_val
        # p2_x = p0_x - 50*sin_val
        # p2_y = p0_y + 50*cos_val
        # p3_x = p0_x + 50*sin_val
        # p3_y = p0_y - 50*cos_val
        # p0_x = int(round(p0_x))
        # p0_y = int(round(p0_y))
        # p1_x = int(round(p1_x))
        # p1_y = int(round(p1_y))
        # p2_x = int(round(p2_x))
        # p2_y = int(round(p2_y))
        # cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        # cv.putText(image, str(confidence), (p0_x, p0_y),
        #            cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        # if marking_point.shape > 0.5:
        #     cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        # else:
        #     p3_x = int(round(p3_x))
        #     p3_y = int(round(p3_y))
        #     cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)
    # img2show.show()
    # pass

def plot_slots(image, pred_points, slots):
    """Plot parking slots on the image."""
    if not pred_points or not slots:
        return
    marking_points = list(list(zip(*pred_points))[1])
    height = image.shape[0]
    width = image.shape[1]
    for slot in slots:
        point_a = marking_points[slot[0]]
        point_b = marking_points[slot[1]]
        p0_x = width * point_a.x - 0.5
        p0_y = height * point_a.y - 0.5
        p1_x = width * point_b.x - 0.5
        p1_y = height * point_b.y - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)
        distance = calc_point_squre_dist(point_a, point_b)
        if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
            separating_length = config.LONG_SEPARATOR_LENGTH
        elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:
            separating_length = config.SHORT_SEPARATOR_LENGTH
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)


def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, thresh, device,isval=False):
    """Given image read from opencv, return detected marking points."""
    if isval:
        prediction = detector(image)
    else:
        prediction = detector(preprocess_image(image).to(device))
    
    # return get_predicted_points_dmpr(prediction[0], thresh)
    return get_predicted_points(prediction, thresh)



def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):
                continue
            result = pair_marking_points(point_i, point_j)
            if result == 1:
                slots.append((i, j))
            elif result == -1:
                slots.append((j, i))
    return slots


def detect_video(detector, device, args):
    """Demo for detecting video."""
    timer = Timer()
    input_video = cv.VideoCapture(args.video)
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    output_video = cv.VideoWriter()
    if args.save:
        output_video.open('record.avi', cv.VideoWriter_fourcc(*'XVID'),
                          input_video.get(cv.CAP_PROP_FPS),
                          (frame_width, frame_height), True)
    frame = np.empty([frame_height, frame_width, 3], dtype=np.uint8)
    while input_video.read(frame)[0]:
        timer.tic()
        pred_points = detect_marking_points(
            detector, frame, args.thresh, device)
        slots = None
        if pred_points and args.inference_slot:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        plot_points(frame, pred_points)
        plot_slots(frame, pred_points, slots)
        cv.imshow('demo', frame)
        cv.waitKey(1)
        if args.save:
            output_video.write(frame)
    print("Average time: ", timer.calc_average_time(), "s.")
    input_video.release()
    output_video.release()


def detect_image(detector, device, args):
    """Demo for detecting images."""
    timer = Timer()
    while True:
        # image_file = input('Enter image file path: ')

        image_file = "/media/fjy/SHARE/dataset/ps2.0/testing/all/0288.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/p2_img115_2256.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/p2_img15_1800.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/p2_img13_1032.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/p2_img8_0186.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/img4_0867.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160725145113_488.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160725142318_088.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160722193621_572.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160722192751_4676.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160722192751_2304.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160722192751_760.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/20161111-05-291.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/20161109-09-13.jpg"
        # image_file = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160725145113_176.jpg"
        image = cv.imread(image_file)
        timer.tic()
        pred_points = detect_marking_points(
            detector, image, args.thresh, device)
        slots = None
        # if pred_points and args.inference_slot:
        #     marking_points = list(list(zip(*pred_points))[1])
        #     slots = inference_slots(marking_points)
        timer.toc()
        slotimg = plot_points(image, pred_points)
        slotimg.show()

        # plot_slots(image, pred_points, slots)
        # cv.imshow('demo', image)
        # cv.waitKey(1)
        if args.save:
            cv.imwrite('save.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 100])


def inference_detector(args):
    """Inference demo of directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)
    # dp_detector = DirectionalPointDetector(
    #     3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    
    dp_detector = get_model().to(device)
    
    dp_detector.load_state_dict(torch.load(args.detector_weights,map_location='cuda:0'))
    dp_detector.eval()
    if args.mode == "image":
        detect_image(dp_detector, device, args)
    elif args.mode == "video":
        detect_video(dp_detector, device, args)


if __name__ == '__main__':
    inference_detector(config.get_parser_for_inference().parse_args())
