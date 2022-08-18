"""Train directional marking point detector."""
import math
import random

import torch
from torch.utils.data import DataLoader
import config
import data
import util
from model import DirectionalPointDetector
from model_slot_det import get_model
from ps_evaluate import psevaluate_train
import numpy as np

# def seed_all(seed):
#     if not seed:
#         seed = 1
#
#     print("[ Using Seed : ", seed, " ]")
#
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# pytorch_random_seed = 1
# seed_all(pytorch_random_seed)

def plot_prediction(logger, image, marking_points, prediction):
    """Plot the ground truth and prediction of a random sample in a batch."""
    rand_sample = random.randint(0, image.size(0)-1)
    sampled_image = util.tensor2im(image[rand_sample])
    logger.plot_marking_points(sampled_image, marking_points[rand_sample],
                               win_name='gt_marking_points')
    sampled_image = util.tensor2im(image[rand_sample])
    pred_points = data.get_predicted_points(prediction[rand_sample], 0.01)
    if pred_points:
        logger.plot_marking_points(sampled_image,
                                   list(list(zip(*pred_points))[1]),
                                   win_name='pred_marking_points')


def generate_objective(marking_points_batch, device):
    """Get regression objective and gradient for directional point detector."""
    batch_size = len(marking_points_batch)
    objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
                            config.FEATURE_MAP_SIZE, config.FEATURE_MAP_SIZE,
                            device=device)
    gradient = torch.zeros_like(objective)
    gradient[:, 0].fill_(0.1)
    for batch_idx, marking_points in enumerate(marking_points_batch):
        for marking_point in marking_points:
            col = math.floor(marking_point.x * config.FEATURE_MAP_SIZE)
            row = math.floor(marking_point.y * config.FEATURE_MAP_SIZE)
            # Confidence Regression
            objective[batch_idx, 0, row, col] = 1.
            # Makring Point Shape Regression
            # objective[batch_idx, 1, row, col] = marking_point.shape
            # Offset Regression
            objective[batch_idx, 1, row, col] = marking_point.x*config.FEATURE_MAP_SIZE - col
            objective[batch_idx, 2, row, col] = marking_point.y*config.FEATURE_MAP_SIZE - row
            # Direction Regression
            objective[batch_idx, 3, row, col] = marking_point.dx
            objective[batch_idx, 4, row, col] = marking_point.dy
            direction = marking_point.direction
            objective[batch_idx, 5, row, col] = math.cos(direction)
            objective[batch_idx, 6, row, col] = math.sin(direction)
            # Assign Gradient
            gradient[batch_idx, 1:7, row, col].fill_(1.)
            gradient[batch_idx, 1:5, row, col].fill_(15.)
    return objective, gradient


def train_detector(args):
    """Train directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    # dp_detector = DirectionalPointDetector(
    #     3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)

    dp_detector = get_model().to(device)
    
    
    if args.detector_weights:
        print("Loading weights: %s" % args.detector_weights)
        dp_detector.load_state_dict(torch.load(args.detector_weights,map_location='cuda:0'))
    # dp_detector.eval()
    dp_detector.train()

    # optimizer = torch.optim.Adam(dp_detector.parameters(), lr=args.lr)

    optimizer = torch.optim.AdamW(dp_detector.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    
    if args.optimizer_weights:
        print("Loading weights: %s" % args.optimizer_weights)
        optimizer.load_state_dict(torch.load(args.optimizer_weights,map_location='cuda:0'))

    logger = util.Logger(args.enable_visdom, ['train_loss'])
    data_loader = DataLoader(data.ParkingSlotDataset(args,is_train=True),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)),
                             pin_memory = True)

    val_loader = DataLoader(data.ParkingSlotDataset(args,is_train=False),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)),
                             pin_memory = True)
    
    psevaluate_train(args, dp_detector, val_loader, device)

    Best_Precision = 0
    Best_epoch = 0
    for epoch_idx in range(args.num_epochs):
        for iter_idx, (images, marking_points) in enumerate(data_loader):
            images = torch.stack(images).to(device)
            # marking_points = marking_points.to(device)

            # pass
            # aa =1
            # continue

            optimizer.zero_grad()
            prediction = dp_detector(images)
            objective, gradient = generate_objective(marking_points, device)
            loss = (prediction - objective) ** 2
            loss.backward(gradient)
            optimizer.step()

            total_loss = torch.sum(loss*gradient).item() / loss.size(0)
            train_loss = torch.sum(loss[:,0,...]*gradient[:,0,...]).item() / loss.size(0)
            dxdy_loss = torch.sum(loss[:,1:3,...]*gradient[:,1:3,...]).item() / loss.size(0)
            xy_loss = torch.sum(loss[:, 3:5, ...] * gradient[:, 3:5, ...]).item() / loss.size(0)
            direction_loss = torch.sum(loss[:, 5:7, ...] * gradient[:, 5:7, ...]).item() / loss.size(0)
            if iter_idx % 20 == 0:
                logger.log(epoch=epoch_idx, iter=iter_idx,total_loss=total_loss, train_loss=train_loss,dxdy_loss=dxdy_loss,xy_loss=xy_loss,direction_loss=direction_loss)
            if args.enable_visdom:
                plot_prediction(logger, images, marking_points, prediction)
        #
        # scheduler.step()
        
        # if epoch_idx % 2 == 0 :
        #     torch.save(dp_detector.state_dict(),
        #                'my_weights/dp_detector_%d.pth' % epoch_idx)
        #     torch.save(optimizer.state_dict(), 'my_weights/optimizer.pth')

        if epoch_idx > 200:
            print("-------starteval------")
            Precision = psevaluate_train(args, dp_detector, val_loader, device)
    
            if Precision > Best_Precision:
                torch.save(dp_detector.state_dict(),
                           'my_weights/dp_detector_%d.pth' % epoch_idx)
                torch.save(optimizer.state_dict(), 'my_weights/optimizer.pth')
                Best_Precision = Precision
                Best_epoch = epoch_idx
            print("Precision: {} | Best_Precision_epoch{}: {}".format(Precision, Best_epoch, Best_Precision))
            dp_detector.train()

if __name__ == '__main__':
    train_detector(config.get_parser_for_training().parse_args())
