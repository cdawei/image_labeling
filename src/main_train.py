import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

if len(sys.argv) != 5:
    print('Usage: ', sys.argv[0], ' WORK_DIR  DATA_INDEX  C  P')
    sys.exit(0)

work_dir = sys.argv[1]
dix = int(sys.argv[2])
C = float(sys.argv[3])
p = float(sys.argv[4])

dataset_names = ['espgame', 'iaprtc12']
src_dir = '%s/src' % work_dir
data_dir = '%s/data/%s' % (work_dir, dataset_names[dix])
sys.path.append(src_dir)

from dataset import ImageDataset
from method import mymodel, criterion_rpc
from util import score2label, compute_metrics, evaluate

batch_size = 128
factor_size = 1024
num_epochs = 50
save = False

print('Dataset: %s, Batch_size: %g, Factor_size: %g, C: %g, p: %g' % (dataset_names[dix], batch_size, factor_size, C, p))
if save:
    fname = 'mymodel_%s_%g_%g_%g_%g_dict.pth' % (dataset_names[dix], batch_size, factor_size, C, p)

loss_func = criterion_rpc
# loss_func = criterion_pcg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

dataset_train = ImageDataset(image_dir='%s/train_image' % data_dir, 
                             image_list_file='%s/train_image_list.txt' % data_dir, 
                             label_file='%s/train_labels.mat' % data_dir, 
                             vocab_file='%s/vocab.txt' % data_dir)

dataset_test = ImageDataset(image_dir='%s/test_image' % data_dir, 
                            image_list_file='%s/test_image_list.txt' % data_dir, 
                            label_file='%s/test_labels.mat' % data_dir, 
                            vocab_file='%s/vocab.txt' % data_dir)

# debug = False
n_batches = int((dataset_train.num_samples - 1) / batch_size)
# progress_num = int((n_batches-1) / 10)
torch.manual_seed(1)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

mm = mymodel(num_labels=dataset_train.num_labels, num_factors=factor_size)
mm = mm.to(device)
optimiser = torch.optim.Adam(mm.parameters(), weight_decay=C)
mm.train()
# losses = []
t0 = time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    # print('-' * 10)

    t1 = time.time()
    # exp_lr_scheduler.step()
    running_loss = 0.0
    
    # if debug:
    #     true_labels_train = np.zeros((dataset_train.num_samples, dataset_train.num_labels))
    #     pred_scores_train = np.zeros(true_labels_train.shape, dtype=np.uint8)
    #     start_ix = 0
    #     end_ix = 0

    # Iterate over data
    # for bix, samples in enumerate(dataloader_train):
    # for samples in tqdm(dataloader_train):
    for samples in dataloader_train:
        inputs = samples['image'].to(device)
        labels = samples['labels'].to(device)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward
        scores = mm(inputs)
        loss = loss_func(labels, scores, p=p)

        # backward + optimise
        loss.backward()
        optimiser.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        # print('Loss (batch {:d}): {:.4f}'.format(bix+1, loss.item()))
        # losses.append(loss.item())
        # if (bix + 1) % progress_num == 0:
        #    sys.stdout.write('-')

        # if debug:
        #     end_ix += inputs.size(0)
        #     pred_scores_train[start_ix:end_ix, :] = scores.cpu().data
        #     true_labels_train[start_ix:end_ix, :] = labels.cpu().numpy()
        #     start_ix = end_ix
       
    epoch_loss = running_loss / dataset_train.num_samples
    print('\nLoss (epoch {:d}): {:.4f}\n'.format(epoch+1, epoch_loss))
    print('Time: %.1f sec' % (time.time() - t1))
    evaluate(dataset_test, mm)
    
    # if debug:
    #     compute_metrics(Y_true=true_labels_train, Y_pred=score2label(pred_scores_train))

time_elapsed = time.time() - t0
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
if save:
    torch.save(mm.state_dict(), fname)
