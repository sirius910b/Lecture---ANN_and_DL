import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import dataset
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):  # LSTM의 경우
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:  # RNN의 경우
            hidden = hidden.to(device)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM의 경우
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:  # RNN의 경우
                hidden = hidden.to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main(model_type):
    input_file = 'shakespeare_train.txt'
    batch_size = 64
    hidden_size = 256
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 10

    dataset_obj = dataset.Shakespeare(input_file)
    dataset_size = len(dataset_obj)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=val_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'RNN':
        model = CharRNN(len(dataset_obj.chars), hidden_size, num_layers).to(device)
    elif model_type == 'LSTM':
        model = CharLSTM(len(dataset_obj.chars), hidden_size, num_layers).to(device)
    else:
        raise ValueError("Invalid model type. Choose either 'RNN' or 'LSTM'.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        trn_loss = train(model, train_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 모델 가중치 저장
    torch.save(model.state_dict(), f'char_{model_type.upper()}.pth')

    # 손실 값 플롯
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} Model Loss')
    plt.legend()
    plt.savefig(f'{model_type}_loss_plot.png')
    plt.show()

if __name__ == '__main__':
    model_type = 'RNN'  # 또는 'RNN'
    main(model_type)

