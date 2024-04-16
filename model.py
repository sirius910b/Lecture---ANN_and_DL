

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        # 첫 번째 합성곱 층: 입력 채널 1개 (흑백 이미지), 출력 채널 6개, 커널 크기 5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 두 번째 합성곱 층: 입력 채널 6개, 출력 채널 16개, 커널 크기 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 첫 번째 완전 연결 층: 입력 특징 256개 (16 * 4 * 4), 출력 특징 120개
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 두 번째 완전 연결 층: 입력 특징 120개, 출력 특징 84개
        self.fc2 = nn.Linear(120, 84)
        # 세 번째 완전 연결 층 (출력 층): 입력 특징 84개, 출력 특징 10개 (숫자 0-9)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        # 첫 번째 합성곱 층 적용 후 ReLU 활성화 함수 및 최대 풀링 적용
        x = F.max_pool2d(F.relu(self.conv1(img)), kernel_size=(2, 2))
        # 두 번째 합성곱 층 적용 후 ReLU 활성화 함수 및 최대 풀링 적용
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        # 완전 연결 층을 위해 텐서 평탄화
        x = x.view(-1, self.num_flat_features(x))
        # 첫 번째 완전 연결 층에 ReLU 적용
        x = F.relu(self.fc1(x))
        # 두 번째 완전 연결 층에 ReLU 적용
        x = F.relu(self.fc2(x))
        # 출력 층
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 배치 차원을 제외한 모든 차원의 크기를 계산
        size = x.size()[1:]  # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """
    def __init__(self):
        super(CustomMLP, self).__init__()
        # 첫 번째 완전 연결 층
        self.fc1 = nn.Linear(28*28, 64)  # 입력 특징 784개 (28x28), 출력 특징 64개
        # 출력 층
        self.fc2 = nn.Linear(64, 10)     # 입력 특징 64개, 출력 특징 10개 (숫자 0-9)

    def forward(self, img):
        # 이미지를 평탄화
        x = img.view(-1, 28*28)
        # 첫 번째 완전 연결 층에 ReLU 활성화 함수 적용
        x = F.relu(self.fc1(x))
        # 출력 층
        x = self.fc2(x)
        return x




# TRY : UPGRADE
class LeNet5_upgrade(nn.Module):
    def __init__(self):
        super(LeNet5_upgrade, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)  # 배치 정규화 추가
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)  # 배치 정규화 추가
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(img))), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=(2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))  # 드롭아웃 적용
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features






