import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime
import json
import argparse
import time
import random
import cal_centers as cc
import generate_vec as gv
import search_vec as sv
import warnings

import matplotlib.pyplot as plt

import matplotlib.lines as lines
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
# Accept multiple target labels as comma-separated string, e.g. --labels 1,3,5
parser.add_argument('--labels', type=str, required=True,
                    help='comma-separated target classes for your attack, e.g. "1,3,5" or single "2"')
parser.add_argument('--dup', type=int, required=True,
                    help='the ID for duplicated models of a same setting')
parser.add_argument('--magnification', type=int, required=True,
                    help='the size of the auxiliary set will be 50*magnification')
parser.add_argument('--multies', type=int, required=False,
                    default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False,
                    default=0.25, help='the feature ratio held by the attacker')
parser.add_argument('--clean-epoch', type=int, required=False,
                    default=80, help='the number of training epochs without poisoning')

args = parser.parse_args()
# parse labels into a list of ints
if ',' in args.labels:
    target_labels = [int(x.strip())
                     for x in args.labels.split(',') if x.strip() != '']
else:
    # allow a single number
    target_labels = [int(args.labels)]

other_unit = (1-args.unit)/(args.multies-1)

target_num = 50
normal_num = 50
clean_epoch = args.clean_epoch


def prepared_data(set):
    data = []
    label = []
    for idx in range(len(set)):
        x, y = set[idx]
        data.append({'id': idx, 'data': x})
        label.append(y)

    return data, label


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        x = self.data[item]['data']
        if not (self.transform is None):
            x = self.transform(x)
        y = self.label[item]
        id = self.data[item]['id']

        return x, y, id

    def __len__(self):
        return len(self.label)


def steal_samples(trn_x, trn_y, t):
    """
    Steal clean examples for label t.
    Returns: (steal_set, steal_id_tensor, steal_id_python_set)
    """
    targets = []
    for idx in range(len(trn_y)):
        if trn_y[idx] == t:
            targets.append(trn_x[idx]['id'])
    num = target_num*target_magnification
    print("clean image used for class %d: %d" % (t, num))

    steal_id = random.sample(targets, num)
    data = []
    label = []
    for idx in steal_id:
        data.append(trn_x[idx])
        label.append(trn_y[idx])

    steal_set = CIFAR10(data, label, transform=transform_for_train)
    steal_id_tensor = torch.tensor(steal_id)
    steal_id_set = set([int(x) for x in steal_id])

    return steal_set, steal_id_tensor, steal_id_set


def design_vec(class_num, model, label, steal_set):
    """
    Same as before: design one vector for one label using the given steal_set.
    """
    target_clean_vecs = gv.generate_target_clean_vecs(
        model.models[0], steal_set, args.unit, bottom_series=0)

    dim = filter_dim(target_clean_vecs)

    center = cc.cal_target_center(
        target_clean_vecs[dim].copy(), kernel_bandwidth=1000)

    target_vec = sv.search_vec(center, target_clean_vecs, args.unit)

    # original reshape kept
    target_vec = target_vec.reshape(
        (64, int((int((int(32*args.unit)-2)/2+1)-2)/2+1), 8))

    return target_vec


def filter_dim(vecs):
    coef = np.corrcoef(vecs)
    rows = np.sum(coef, axis=1)
    selected = np.argpartition(rows, -target_num)[-target_num:]
    print(np.mean(np.corrcoef(vecs[selected])))
    return selected


def train_model(model, dataloader, target_labels, steal_sets_map, steal_id_sets_map,
                epoch_num, start_epoch=0, is_binary=False, verbose=True):
    """
    Modified to accept:
      - target_labels: list of labels to attack
      - steal_sets_map: dict label -> steal_set (for design_vec)
      - steal_id_sets_map: dict label -> set(ids) (for fast membership testing)
    During poisoning epochs we maintain a dict of vec_arrs (one per label).
    When a training sample id is in steal_id_sets_map[label], we replace the batch feature
    with the corresponding vec_arr for that label.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # initialize vecs dict (None until constructed)
    vecs_map = {t: None for t in target_labels}

    for epoch in range(start_epoch, epoch_num):
        t1 = time.time()

        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0

        # update vectors for every target label periodically (same schedule as before)
        if epoch >= clean_epoch and epoch % 10 == 0:
            for t in target_labels:
                print("Designing vector for label", t, "at epoch", epoch)
                vecs_map[t] = design_vec(
                    class_num, model, t, steal_sets_map[t])
                # ensure CUDA tensor stored for quick assignment later
                vecs_map[t] = torch.tensor(vecs_map[t]).float().cuda()

        for i, (x_in, y_in, id_in) in enumerate(dataloader):
            B = x_in.size()[0]

            if args.unit != 1:
                x_list = x_in.split([int(x_in.size()[2]*args.unit)]+[int(x_in.size()[2]*other_unit) for _ in range(args.multies-2)]+[
                                    x_in.size()[2]-int(x_in.size()[2]*args.unit)-(args.multies-2)*int(x_in.size()[2]*other_unit)], dim=2)
            else:
                x_list = [x_in]

            vec1 = model.models[0](x_list[0])

            # Poison stolen training samples after clean_epoch: may need to inject different vecs based on label membership
            if epoch >= clean_epoch:
                # convert id_in to python ints for membership checks
                id_in_list = [int(x) for x in id_in]
                # For each target label, find positions in the batch that are stolen for that label
                for t in target_labels:
                    if vecs_map[t] is None:
                        continue  # vector not designed yet
                    # find positions within batch where id belongs to this steal set
                    positions = [idx for idx, did in enumerate(
                        id_in_list) if did in steal_id_sets_map[t]]
                    if len(positions) == 0:
                        continue
                    # assign vec for these positions
                    # vec1 shape: (B, C, H, W?) - we assign with advanced indexing
                    vec1[positions] = vecs_map[t]

            if args.unit != 1:
                vec = torch.cat([vec1]+[model.models[i](x_list[i])
                                for i in range(1, args.multies)], dim=2)
            else:
                vec = vec1

            pred = model.top(vec)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B

        if verbose:
            t2 = time.time()
            print("Epoch %d, loss = %.4f, acc = %.4f (%.4fs)" %
                  (epoch, cum_loss / tot, cum_acc / tot, t2-t1))

    # return last-designed vecs_map (may contain None for labels whose vec wasn't designed)
    return vecs_map


def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in, _) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot


def attack_model(model, dataloader, vecs_map, target_labels, multies, unit, other_unit):
    """
    For each target label t in target_labels, force vecs_map[t] into the attacker bottom-model
    (repeat for whole batch) and compute attack success for that label.
    Returns dict label -> attack_accuracy.
    """
    model.eval()
    # prepare accumulators per target
    acc_map = {t: 0.0 for t in target_labels}
    tot = 0.0

    # iterate over dataset once and for each batch test each target vector
    for i, (x_in, y_in, _) in enumerate(dataloader):
        B = x_in.size()[0]
        tot += B

        if args.unit != 1:
            x_list = x_in.split([int(x_in.size()[2]*unit)]+[int(x_in.size()[2]*other_unit) for _ in range(multies-2)]+[
                                x_in.size()[2]-int(x_in.size()[2]*unit)-(multies-2)*int(x_in.size()[2]*other_unit)], dim=2)

        # for each target, build vec where bottom part is repeated vec and others are real features
        for t in target_labels:
            if vecs_map.get(t) is None:
                # no vector designed for this target (e.g. never created during training)
                continue
            # repeat vector across batch (ensure same dtype/device)
            # assume vec tensors are (C,H,W) shaped
            vec1 = vecs_map[t].unsqueeze(0).repeat(B, 1, 1, 1)
            vec1 = vec1.cuda()

            if args.unit != 1:
                # compute other parts features from model (non-attacker parts)
                other_parts = [model.models[i](x_list[i])
                               for i in range(1, multies)]
                vec = torch.cat([vec1] + other_parts, dim=2)
            else:
                vec = vec1

            pred = model.top(vec)
            pred_c = pred.max(1)[1].cpu()
            # compare prediction to the forced target label
            acc_map[t] += (pred_c.eq(torch.full((B,), t,
                           dtype=torch.long))).sum().item()

    # normalize
    for t in list(acc_map.keys()):
        acc_map[t] = acc_map[t] / tot

    return acc_map


if __name__ == '__main__':
    target_magnification = args.magnification

    GPU = True
    if GPU:
        torch.cuda.manual_seed_all(args.dup)
        random.seed(args.dup)
        torch.manual_seed(args.dup)
        np.random.seed(args.dup)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE = 500
    N_EPOCH = 100
    transform_for_train = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=5),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    transform_for_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./raw_data/', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(
        root='./raw_data/', train=False, download=True)

    trn_x, trn_y = prepared_data(trainset)
    dl_train_set = CIFAR10(trn_x, trn_y, transform=transform_for_train)
    val_x, val_y = prepared_data(testset)
    dl_val_set = CIFAR10(val_x, val_y, transform=transform_for_test)
    is_binary = False
    need_pad = False

    from model_split import Model

    input_size = (3, 32, 32)
    class_num = 10

    model = Model(gpu=GPU, multies=args.multies, unit=args.unit)
    trainloader = torch.utils.data.DataLoader(
        dl_train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        dl_val_set, batch_size=BATCH_SIZE, shuffle=True)

    # For multi-target: build steal_set and id_set for each target label
    steal_sets_map = {}
    steal_id_tensors_map = {}
    steal_id_sets_map = {}
    for t in target_labels:
        sset, sids_tensor, sids_set = steal_samples(trn_x, trn_y, t)
        steal_sets_map[t] = sset
        steal_id_tensors_map[t] = sids_tensor
        steal_id_sets_map[t] = sids_set

    dup = args.dup

    t1 = time.time()

    model.load_state_dict(torch.load('clean-%d-%d-%s.model' %
                          (args.dup, args.multies, args.unit)))

    # train_model now returns a dict label -> vec_tensor (on cuda)
    last_vecs_map = train_model(model, trainloader, target_labels,
                                steal_sets_map, steal_id_sets_map,
                                epoch_num=N_EPOCH, start_epoch=clean_epoch, is_binary=is_binary, verbose=True)

    # save model (same naming idea: include labels in filename)
    labels_str = "_".join([str(l) for l in target_labels])
    torch.save(model.state_dict(
    ), f'poison_labels_{args.dup}-{args.multies}-{args.unit}-{labels_str}.model')

    cleanacc = eval_model(model, testloader, is_binary=is_binary)
    print('clean acc: %.4f' % cleanacc)

    # attack_model returns dict label -> attack accuracy
    atk_acc_map = attack_model(model, testloader, last_vecs_map,
                               target_labels, args.multies, args.unit, other_unit)
    for t, a in atk_acc_map.items():
        print(f'target label: {t}, attack acc: {a:.4f}')

    # save vectors where available (convert to numpy before saving)
    for t, vec in last_vecs_map.items():
        if vec is None:
            continue
        # vec is a cuda tensor (C,H,W). move to cpu numpy
        np.save(
            f'label_{args.dup}-{args.multies}-{args.unit}-{t}_vec.npy', vec.cpu().numpy())

    t2 = time.time()
    print("Training a model costs %.4fs." % (t2-t1))
