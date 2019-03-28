import sys
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def show_batch(sample_batched, label_decoder=None):
    """Show image with labels for a batch of samples."""
    images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)

    grid = torchvision.utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    if label_decoder is not None:
        assert callable(label_decoder)
        for i in range(batch_size):
            print('%d, labels:' % i, label_decoder(labels_batch[i]))
    plt.title('Batch from dataloader')


def score2label(scores, topk=5):
    assert len(scores.shape) == 2
    if type(scores) != torch.Tensor:
        assert type(scores) == np.ndarray
        scores = torch.tensor(scores)
    batch_size, num_labels = scores.shape
    _, topix = torch.topk(scores, k=topk, dim=1)
    row_ix = torch.arange(batch_size).reshape(-1, 1).expand(-1, topk).reshape(-1)
    col_ix = topix.reshape(-1)
    assert row_ix.shape == col_ix.shape
    preds = torch.zeros(scores.shape, dtype=torch.uint8)
    preds[row_ix, col_ix] = 1
    return preds


def compute_metrics(Y_true, Y_pred):
    if type(Y_true) == torch.Tensor:
        Y_true = Y_true.numpy()
    if type(Y_pred) == torch.Tensor:
        Y_pred = Y_pred.numpy()
    assert len(Y_true.shape) == 2
    assert Y_true.shape == Y_pred.shape
    p, r, _, _ = precision_recall_fscore_support(Y_true, Y_pred, beta=1.0, average=None)

    # evaluation as those in previous works
    mp = np.mean(p)
    mr = np.mean(r)
    f1 = 2 * mp * mr / (mp + mr)
    Nplus = np.sum(r > 0)

    print('P: {:.0f} | R: {:.0f} | F1: {:.0f} | N+: {:d}'.format(mp * 100, mr * 100, f1 * 100, Nplus))
    # with open('evaluate.log', 'a') as fd:
    #     fd.write('P: {:.0f} | R: {:.0f} | F1: {:.0f} | N+: {:d}\n'.format(mp * 100, mr * 100, f1 * 100, Nplus))


def evaluate(dataset, model):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    pred_scores = np.zeros((dataset.num_samples, dataset.num_labels))
    true_labels = np.zeros_like(pred_scores)
    start_ix = 0
    end_ix = 0
    model.eval()
    # device = model.lookup_tensor.device
    for batch_test in dataloader:
        images = batch_test['image']
        if model.gpu_device:
            images = images.to(model.gpu_device)
        labels = batch_test['labels']
        scores = model(images)
        end_ix += images.size(0)
        pred_scores[start_ix:end_ix, :] = scores.cpu().data
        true_labels[start_ix:end_ix, :] = labels.numpy()
        start_ix = end_ix
        # sys.stdout.write('.')
    # print()
    compute_metrics(Y_true=true_labels, Y_pred=score2label(pred_scores))

