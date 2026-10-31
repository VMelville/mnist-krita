# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.5.4", "torch==2.14.1"]
# ///
"""Train 784 -> 72 ReLU -> 10 on MNIST; no fixed random seed.

uv run train.py
Output: model.npz (local generated weights; excluded from distribution).
"""
import gzip
import hashlib
import json
import platform
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.cache' / 'mnist'
SOURCE = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
         'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
         't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
         't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c'}


def read_idx(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print('Downloading ' + name, flush=True)
        urllib.request.urlretrieve(SOURCE + name, path)
    if hashlib.md5(path.read_bytes()).hexdigest() != FILES[name]:
        raise ValueError('MNIST checksum mismatch: ' + str(path))
    raw = gzip.decompress(path.read_bytes())
    zero, kind, ndim = struct.unpack('>HBB', raw[:4])
    if zero != 0 or kind != 8:
        raise ValueError('Expected uint8 IDX file')
    shape = struct.unpack('>' + 'I' * ndim, raw[4:4 + ndim*4])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + ndim*4).reshape(shape).copy()


class Network(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.fc1 = nn.Linear(784, 72)
        self.fc2 = nn.Linear(72, 10)
        self.dropout = dropout

    def forward(self, x):
        h = self.fc1(x.flatten(1)).relu()
        return self.fc2(F.dropout(h, p=self.dropout, training=self.training))


def shift(x, radius):
    if radius == 0:
        return x
    padded = F.pad(x, (radius, radius, radius, radius))
    origins = torch.randint(0, 2*radius + 1, (len(x), 2))
    ys = origins[:, 0, None, None] + torch.arange(28)[None, :, None]
    xs = origins[:, 1, None, None] + torch.arange(28)[None, None, :]
    return padded[torch.arange(len(x))[:, None, None], ys, xs]


@torch.inference_mode()
def evaluate(model, x, y):
    model.eval()
    correct, loss = 0, 0.
    for start in range(0, len(x), 1024):
        logits = model(x[start:start+1024])
        target = y[start:start+1024]
        correct += int((logits.argmax(1) == target).sum())
        loss += F.cross_entropy(logits, target, reduction='sum').item()
    return {'count': len(x), 'correct': correct, 'accuracy': correct/len(x), 'loss': loss/len(x)}


def main():
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(config['threads'])
    images, labels = read_idx('train-images-idx3-ubyte.gz'), read_idx('train-labels-idx1-ubyte.gz')
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise ValueError('Unexpected MNIST training shape')
    # Neither the split nor initialization/augmentation/shuffling has a fixed seed.
    order = np.random.default_rng().permutation(len(images))
    cut = len(images) - config['validation_count']
    if not 0 < cut < len(images):
        raise ValueError('Invalid validation_count')
    train_ids, validation_ids = order[:cut], order[cut:]
    np.savez_compressed(CACHE.parent/'split.npz', train=train_ids, validation=validation_ids)
    x, y = torch.from_numpy(images).float()/255, torch.from_numpy(labels).long()
    xt, yt = x[train_ids], y[train_ids]
    xv, yv = x[validation_ids], y[validation_ids]
    model = Network(config['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['minimum_learning_rate'])
    best_key, best_state, best_epoch, history = None, None, None, []
    for epoch in range(1, config['epochs']+1):
        model.train()
        for ids in torch.randperm(len(xt)).split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(shift(xt[ids], config['shift_radius'])), yt[ids])
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation = evaluate(model, xv, yv)
        history.append({'epoch': epoch, **validation})
        key = (validation['accuracy'], -validation['loss'])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        print(f"epoch {epoch:02d}: validation {validation['accuracy']:.2%}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    # Load test data only after freezing the validation-selected checkpoint.
    test_images, test_labels = read_idx('t10k-images-idx3-ubyte.gz'), read_idx('t10k-labels-idx1-ubyte.gz')
    if test_images.shape != (10000, 28, 28) or test_labels.shape != (10000,):
        raise ValueError('Unexpected MNIST test shape')
    test_x, test_y = torch.from_numpy(test_images).float()/255, torch.from_numpy(test_labels).long()
    test = evaluate(model, test_x, test_y)
    with torch.inference_mode():
        expected = model(test_x[:1])[0].numpy()
    weights = {name.replace('.', '_'): value.numpy() for name, value in best_state.items()}
    np.savez_compressed(ROOT/'model.npz', **weights,
                        initial_input=test_x[0].flatten().numpy(), expected_logits=expected)
    report = {'config': config, 'fixed_seed': False, 'selected_epoch': best_epoch,
              'validation': history[best_epoch-1], 'test': test, 'history': history,
              'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__}
    (CACHE.parent/'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Saved model.npz; test {test['correct']}/{test['count']} = {test['accuracy']:.2%}")


if __name__ == '__main__':
    main()
