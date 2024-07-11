import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    seq_before_resi=nn.Sequential(
        nn.Linear(dim,hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim,dim),
        norm(dim)
    )
    seq=nn.Sequential(nn.Residual(seq_before_resi),nn.ReLU())
    return seq
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules=[nn.Linear(dim,hidden_dim),nn.ReLU()]
    for i in range(num_blocks):
        modules.append(ResidualBlock(dim=hidden_dim,hidden_dim=hidden_dim//2,norm=norm,drop_prob=drop_prob))
    modules.append(nn.Linear(hidden_dim,num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    loss_f=nn.SoftmaxLoss()
    
    loss_sum=0
    error_sum=0
    size=0

    for idx,(images,labels) in enumerate(dataloader):
        logits=model(images)
        loss=loss_f(logits,labels)

        if opt: 
            loss.backward()
            opt.step()
            opt.reset_grad()

        predicted_labels = np.argmax(logits.numpy(), axis=1)
        error_sum+=(predicted_labels!=labels.numpy()).sum()
        loss_sum+=loss.numpy()
        size+=images.shape[0]

    return error_sum/size,loss_sum/(idx+1)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tran_dataset=ndl.data.MNISTDataset(
        data_dir+"/train-images-idx3-ubyte.gz",
        data_dir+"/train-labels-idx1-ubyte.gz"
    )
    train_dataloader=ndl.data.DataLoader(tran_dataset,batch_size=batch_size,shuffle=True)
    test_dataset=ndl.data.MNISTDataset(
        data_dir+"/t10k-images-idx3-ubyte.gz",
        data_dir+"/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader=ndl.data.DataLoader(test_dataset,batch_size=batch_size)

    model=MLPResNet(dim=784,hidden_dim=hidden_dim)
    optimizer=optimizer(params=model.parameters(),lr=lr,weight_decay=weight_decay)

    for i in range(epochs):
        train_err,train_loss=epoch(train_dataloader,model,optimizer)
        print("ecpoch:{} averge_error:{}  averge_error:{}".format(i,train_err,train_loss))

    test_acc,test_loss=epoch(test_dataloader,model)
    return train_err,train_loss,test_acc,test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
