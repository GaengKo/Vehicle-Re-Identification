# %%
#%load_ext autoreload
#%autoreload 2

# %%
"""
# Experiments1
We'll go through learning feature embeddings using different loss functions on MNIST dataset. This is just for visualization purposes, thus we'll be using 2-dimensional embeddings which isn't the best choice in practice.

For every experiment the same embedding network is used (32 conv 5x5 -> PReLU -> MaxPool 2x2 -> 64 conv 5x5 -> PReLU -> MaxPool 2x2 -> Dense 256 -> PReLU -> Dense 256 -> PReLU -> Dense 2) and we don't do any hyperparameter search.
"""

# %%
"""
# Prepare dataset
We'll be working on MNIST dataset
"""


# %%
#from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import BalancedBatchSampler

Veri_transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mean, std = 0.1307, 0.3081
train_dataset = ImageFolder('../data/conferece_dataset/train/train',transform=Veri_transform)
test_dataset = ImageFolder('../data/conferece_dataset/train/test',transform=Veri_transform)
#n_classes = 10
#print(train_dataset.targets)
"""
m_train_dataset = MNIST('../data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
print(train_dataset.classes)
m_test_dataset = MNIST('../data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
"""

#print(train_dataset.classes)
# %%
"""
## Common setup
"""

# %%
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

#matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

mnist_classes = train_dataset.classes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None,train=True):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==(9-i))[0]
        #print(inds)
        c = colors[9-i]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha =0.5 ,color=c)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    if train:
        plt.savefig('./train_result.png',dpi=300)
    else:
        plt.savefig('./test_result.png', dpi=300)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            #images = transform.resize(io.imread(images), (224, 224))
            #images = Image.fromarray(images, mode='RGB')
            #print(len(images))
            #print(images.shape)
            if cuda:
                #images = torch.tensor(images)
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            #print(target)
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels



# %%
"""
While the embeddings look separable (which is what we trained them for), they don't have good metric properties. They might not be the best choice as a descriptor for new classes.



# Triplet network
We'll train a triplet network, that takes an anchor, positive (same class as anchor) and negative (different class than anchor) examples. The objective is to learn embeddings such that the anchor is closer to the positive example than it is to the negative example by some margin value.

![alt text](images/anchor_negative_positive.png "Source: FaceNet")
Source: [2] *Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering.](https://arxiv.org/abs/1503.03832) CVPR 2015.*

**Triplet loss**:   $L_{triplet}(x_a, x_p, x_n) = max(0, m +  \lVert f(x_a)-f(x_p)\rVert_2^2 - \lVert f(x_a)-f(x_n)\rVert_2^2$\)
"""

# %%
# Set up data loaders
#from datasets import Triplet_Veri
#from datasets import TripletMNIST
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric
#triplet_train_dataset = Triplet_Veri(train_dataset,True) # Returns triplets of images
#triplet_test_dataset = Triplet_Veri(test_dataset, False)
import os
train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=50, n_samples=7)
test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=50, n_samples=7)
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
#triplet_m_train_dataset = TripletMNIST(m_train_dataset) # Returns triplets of images
#triplet_m_test_dataset = TripletMNIST(m_test_dataset)
#torch.Size(np.asarray(triplet_train_dataset[0]))
#batch_size = 150
#triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
#triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
#print(type(triplet_test_loader))

#triplet_m_train_loader = torch.utils.data.DataLoader(triplet_m_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
#triplet_m_test_loader = torch.utils.data.DataLoader(triplet_m_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from kittiNet import EmbeddingNet, TripletNet, Net
from losses import TripletLoss

margin = 1

embedding_net = Net()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer,2, gamma=0.9, last_epoch=-1)
n_epochs = 100
log_interval = 1

if os.path.isfile('./model/VeRi_checkpoint'):
    print('*load Data ㅋㅋ*')
    checkpoint = torch.load('./model/VeRi_checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
"""
torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
            }, './model/210408_checkpoint')
"""
# %%
fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

torch.save({
            'model_state_dict': model.state_dict()
            }, './model/RIAM_finetuning_final_checkpoint')

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

# %%
#train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
#print(len(train_labels_tl))
#plot_embeddings(train_embeddings_tl, train_labels_tl)
#val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
#plot_embeddings(val_embeddings_tl, val_labels_tl)