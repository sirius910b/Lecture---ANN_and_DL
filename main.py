

# import some packages you need here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
from model import LeNet5, CustomMLP, LeNet5_upgrade

import matplotlib.pyplot as plt


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.train()  # 모델을 훈련 모드로 설정
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 기울기 초기화
        output = model(data)  # 데이터를 모델에 전달하여 출력 얻기
        loss = criterion(output, target)  # 손실 계산
        loss.backward()  # 손실을 기준으로 역전파 실행
        optimizer.step()  # 매개변수 업데이트
        
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    trn_loss = running_loss / total
    acc = 100. * correct / total
    return trn_loss, acc



def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 기울기 계산을 비활성화
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    tst_loss = test_loss / total
    acc = 100. * correct / total
    return tst_loss, acc

    return tst_loss, acc




global lenet_trn_losses, lenet_trn_accs, lenet_tst_losses, lenet_tst_accs
global mlp_trn_losses, mlp_trn_accs, mlp_tst_losses, mlp_tst_accs

lenet_epoch = 10
mlp_epoch = 10

import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터셋 로드 및 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 모델, 최적화기, 손실 함수 초기화 (LeNet5와 CustomMLP 두 모델 모두)
    model_lenet = LeNet5().to(device)
    # model_lenet = LeNet5_upgrade().to(device) # Upgrade 버전
    model_mlp = CustomMLP().to(device)
    
    optimizer_lenet = optim.SGD(model_lenet.parameters(), lr=0.01, momentum=0.9)
    optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=0.01, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    # 성능 기록을 위한 리스트 초기화
    global lenet_trn_losses, lenet_trn_accs, lenet_tst_losses, lenet_tst_accs
    global mlp_trn_losses, mlp_trn_accs, mlp_tst_losses, mlp_tst_accs
    lenet_trn_losses, lenet_trn_accs, lenet_tst_losses, lenet_tst_accs = [], [], [], []
    mlp_trn_losses, mlp_trn_accs, mlp_tst_losses, mlp_tst_accs = [], [], [], []
    
    # 훈련 및 테스트 (LeNet5)
    for epoch in range(lenet_epoch):  # 에포크 수 설정
        trn_loss, trn_acc = train(model_lenet, train_loader, device, criterion, optimizer_lenet)
        tst_loss, tst_acc = test(model_lenet, test_loader, device, criterion)
        
        lenet_trn_losses.append(trn_loss)
        lenet_trn_accs.append(trn_acc)
        lenet_tst_losses.append(tst_loss)
        lenet_tst_accs.append(tst_acc)
        
        print(f'LeNet-5 Epoch {epoch+1} Train Loss: {trn_loss:.4f} Acc: {trn_acc:.2f}% Test Loss: {tst_loss:.4f} Acc: {tst_acc:.2f}%')

    # 훈련 및 테스트 (CustomMLP)
    for epoch in range(mlp_epoch):  # 에포크 수 설정
        trn_loss, trn_acc = train(model_mlp, train_loader, device, criterion, optimizer_mlp)
        tst_loss, tst_acc = test(model_mlp, test_loader, device, criterion)
        
        mlp_trn_losses.append(trn_loss)
        mlp_trn_accs.append(trn_acc)
        mlp_tst_losses.append(tst_loss)
        mlp_tst_accs.append(tst_acc)
        
        print(f'CustomMLP Epoch {epoch+1} Train Loss: {trn_loss:.4f} Acc: {trn_acc:.2f}% Test Loss: {tst_loss:.4f} Acc: {tst_acc:.2f}%')

    # 결과 그래프 출력
    epochs = range(1, 11)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, lenet_trn_losses, 'r-', label='LeNet-5 Training Loss')
    plt.plot(epochs, mlp_trn_losses, 'b-', label='CustomMLP Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, lenet_trn_accs, 'r-', label='LeNet-5 Training Accuracy')
    plt.plot(epochs, mlp_trn_accs, 'b-', label='CustomMLP Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, lenet_tst_losses, 'r-', label='LeNet-5 Test Loss')
    plt.plot(epochs, mlp_tst_losses, 'b-', label='CustomMLP Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, lenet_tst_accs, 'r-', label='LeNet-5 Test Accuracy')
    plt.plot(epochs, mlp_tst_accs, 'b-', label='CustomMLP Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
    
    lenet_tst_accs_avg = sum(lenet_tst_accs) / len(lenet_tst_accs)
    mlp_tst_accs_avg = sum(mlp_tst_accs) / len(mlp_tst_accs)
    
    print(f"LeNet-5의 예측 성능 : {lenet_tst_accs_avg}%")
    print(f"사용자 정의MLP의 예측 성능 : {mlp_tst_accs_avg}%")
    

    


