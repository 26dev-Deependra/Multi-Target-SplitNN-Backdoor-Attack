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

# Force use of GPU device 0 (CUDA_VISIBLE_DEVICES indexing)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--multies', type=int, required=False,
                    default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False,
                    default=0.25, help='the feature ratio held by the attacker')

args = parser.parse_args()
# 'unit' is the fraction of the input feature width that the attacker holds.
# 'multies' is the number of participants in VFL (vertical federated learning).
unit = args.unit
multies = args.multies
# compute other participants' feature share so total sums to 1
other_unit = (1 - args.unit) / (args.multies - 1)


# -------------------------
# Noise functions
# -------------------------
def add_noise(vec, normal_vecs):
    """
    Noise scheme for two-split VFL (attacker + one honest party).
    - vec: attacker feature vector for a *batch* (tensor)
    - normal_vecs: features coming from the honest participant, used to
                   compute per-dimension average and zero-out low-avg dims.
    Returns the attacked/noised version of 'vec'.
    """
    # compute average across the batch (dim=0) and flatten to 1D
    avg_value = torch.mean(normal_vecs, dim=0).reshape((-1))
    # find indices (columns) where average is very small (< 0.001)
    # these channels will be zeroed-out later
    con = torch.where(avg_value < 0.001)[0]

    size = vec.size()            # save original shape
    vec = vec.reshape((-1))     # flatten to 1D for noise operations

    # clamp to reasonable range and amplify slightly
    vec = vec.clamp_(0, 2.5)
    vec *= 1.15

    # generate two different Gaussian noises (big and small)
    gauss_noise_big = torch.normal(mean=0, std=0.5, size=vec.size()).cuda()
    gauss_noise_small = torch.normal(mean=0, std=0.1, size=vec.size()).cuda()

    # random condition per-element to decide whether to replace with small-noise
    condition = torch.randn(vec.size()).cuda()
    zeros = torch.zeros_like(vec).cuda()

    # 'replace' is either zero (80% prob) or vec + small noise (20% prob)
    replace = torch.where(condition < 0.8, zeros, vec + gauss_noise_small)

    # if element is small (< 0.4) use replace (which often is zero),
    # otherwise add big gaussian noise.
    vec = torch.where(vec < 0.4, replace, vec + gauss_noise_big)

    # clamp negative values to 0, restore batch-dim shape and zero-out low-avg channels
    vec = vec.clamp_(0).reshape((size[0], -1))
    vec[:, con] = 0

    return vec.reshape(size)


def add_noise_multi(vec, normal_vecs):
    """
    Similar to add_noise but with different noise magnitudes intended for
    multi-party (more than 2 participants). Uses smaller stddevs.
    """
    avg_value = torch.mean(normal_vecs, dim=0).reshape((-1))
    con = torch.where(avg_value < 0.001)[0]

    size = vec.size()
    vec = vec.reshape((-1))

    vec = vec.clamp_(0, 2.5)
    vec *= 1.15

    # smaller noise because multi-party setting may be more sensitive
    gauss_noise_big = torch.normal(mean=0, std=0.2, size=vec.size()).cuda()
    gauss_noise_small = torch.normal(mean=0, std=0.05, size=vec.size()).cuda()

    condition = torch.randn(vec.size()).cuda()
    zeros = torch.zeros_like(vec).cuda()
    replace = torch.where(condition < 0.8, zeros, vec + gauss_noise_small)
    vec = torch.where(vec < 0.4, replace, vec + gauss_noise_big)
    vec = vec.clamp_(0).reshape((size[0], -1))
    vec[:, con] = 0

    return vec.reshape(size)


# -------------------------
# Utility: save vectors to CSV for inspection
# -------------------------
def save(vecs, label, normal=False):
    """
    Save first 20 rows of vectors to CSV for debugging/inspection.
    - vecs: tensor of shape (batch, feature_dim, ...)
    - label: integer label used in filename
    - normal: if True write to 'normal_vec_label.csv' else 'noise_vec_label.csv'
    NOTE: the reshape below flattens vectors to match a specific expected shape
    (here using 64*4*8). That number is model-specific — ensure it matches your model.
    """
    vecs = vecs.reshape(-1, 64*4*8)
    if normal:
        f = open('normal_vec_%d.csv' % label, 'w')
    else:
        f = open('noise_vec_%d.csv' % label, 'w')
    # write first 20 rows (or fewer if vecs has less rows implicitly)
    for i in range(20):
        for j in range(vecs.shape[1]):
            f.write(str(vecs[i][j].item()))
            f.write(',')
        f.write('\n')
    f.close()


# -------------------------
# Attack runner
# -------------------------
def attack_model(model, dataloader, vec_arr, label):
    """
    Run the attack on the provided model using the dataset in 'dataloader'.
    - model: instance of your split model (contains .models and .top)
    - dataloader: testloader providing (images, labels)
    - vec_arr: numpy array representing the attacker's target vector (single example)
    - label: the target label the attacker wants to force predictions to
    Returns attack success rate (accuracy vs target label) on the dataloader.
    """
    model.eval()
    cum_acc = 0.0
    tot = 0.0

    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]  # current batch size

        # Repeat the attacker vector to match the batch size and move to GPU
        vec1 = torch.Tensor(np.repeat([vec_arr], B, axis=0)).cuda()

        # SPLIT INPUT: split the input along width (dim=2) into participant shares.
        # The split sizes are:
        #  - attacker: int(x_in.size()[2] * unit)
        #  - each other party: int(x_in.size()[2] * other_unit) (repeated multies-2 times)
        #  - last piece: remainder to ensure sum equals original width
        x_list = x_in.split(
            [int(x_in.size()[2]*unit)] +
            [int(x_in.size()[2]*other_unit) for i in range(multies-2)] +
            [x_in.size()[2] - int(x_in.size()[2]*unit) -
             (multies-2)*int(x_in.size()[2]*other_unit)],
            dim=2
        )

        # Pass the attacker's (honest-looking) share through the attacker's local model
        # 'model.models' is assumed to be a list of local feature extractors for each party.
        vec_normal = model.models[0](x_list[0])

        # Apply noise (attack manipulation) to vec1 using either two-party or multi-party scheme.
        # vec_normal[:20] is used to compute low-average channels -- the code uses
        # only the first 20 examples to estimate avg; this is somewhat arbitrary.
        if multies == 2:
            vec1 = add_noise(vec1, vec_normal[:20])
        elif multies > 2:
            vec1 = add_noise_multi(vec1, vec_normal[:20])

        # Concatenate the (possibly attacked) attacker vector with the honest parties' vectors
        # Note: for parties 1..multies-1 we compute model.models[i](x_list[i])
        vec = torch.cat(
            [vec1] + [model.models[i](x_list[i]) for i in range(1, multies)],
            dim=2  # concatenation along feature dimension
        )

        # pass concatenated features through the top aggregator/classifier
        pred = model.top(vec)

        # predicted class indices
        pred_c = pred.max(1)[1].cpu()

        # compare predicted class to the attack target label (we repeat label across batch)
        cum_acc += (pred_c.eq(torch.Tensor(
            np.repeat([label], B, axis=0)))).sum().item()
        tot = tot + B

    # Save example noised vectors and normal vectors for inspection
    save(vec1.clone().detach().cpu(), label, False)
    save(vec_normal.clone().detach().cpu(), label, True)

    return cum_acc / tot


# -------------------------
# Main: load dataset, create model, iterate targets
# -------------------------
if __name__ == '__main__':

    GPU = True
    if GPU:
        # deterministic GPU behavior for reproducibility
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE = 500
    N_EPOCH = 100

    # standard CIFAR-10 normalization for test images
    transform_for_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    # load CIFAR-10 test set
    testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True,
                                           transform=transform_for_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True)

    # flags possibly used by model (not used further in this script)
    is_binary = False
    need_pad = False

    # import the Model class defined in model_split.py (user's code)
    from model_split import Model

    input_size = (3, 32, 32)
    class_num = 10

    # instantiate the split model with given multies and unit settings
    model = Model(gpu=GPU, multies=multies, unit=unit)

    # Loop over each target label (0..9 for CIFAR-10)
    for label in range(class_num):
        atk_list = []
        # For each 'dup' checkpoint, load model weights and the corresponding target vector
        # then run attack_model and record accuracy. This computes average attack success
        # across 10 different duplicates/checkpoints.
        for dup in range(10):
            # load saved model weights; filename format depends on dup, multies, unit, and label
            model.load_state_dict(torch.load(
                'poison_label_%d-%s-%s-%d.model' % (dup, multies, unit, label)))
            # load the precomputed target vector for this setup (numpy array)
            target_vec = np.load('label_%d-%s-%s-%d_vec.npy' %
                                 (dup, multies, unit, label))
            # run the attack on the test set and get attack accuracy
            atkacc = attack_model(model, testloader, target_vec, label)
            atk_list.append(atkacc)

        # print average attack accuracy across the 10 duplicates
        print('target label: %d, average atk acc: %.4f' %
              (label, sum(atk_list)/len(atk_list)))
