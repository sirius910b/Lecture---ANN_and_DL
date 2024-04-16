
# import some packages you need here
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Normalize


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """


    def __init__(self, data_dir):
        # write your codes here
        self.data_dir = data_dir
        self.img_files = os.listdir(data_dir)
        self.transform = Compose([
            ToTensor(),  
            Normalize((0.1307,), (0.3081,)) # 정규화
        ])


    def __len__(self):
        # write your codes here
        return len(self.img_files)


    def __getitem__(self, idx):
        # write your codes here
        file_name = self.img_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        image = read_image(file_path)
        label = int(file_name.split('_')[1].split('.')[0]) # 레이블 추출
    
        if self.transform:
            image = self.transform(image)
    
        return image, label

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        image = read_image(file_path)  # 여기서 image는 텐서로 반환됩니다.
        label = int(file_name.split('_')[1].split('.')[0])
    
        if self.transform:
            image = self.transform(image)
    
        return image, label


if __name__ == '__main__':
    # Test codes to verify the implementation
    # 경로 설정
    train_data_dir = r'C:\Users\USER\Desktop\임시\인공신경망과 딥러닝\태겸\data\train'
    test_data_dir = r'C:\Users\USER\Desktop\임시\인공신경망과 딥러닝\태겸\data\test'
    
    # MNIST 데이터셋 인스턴스 생성
    train_dataset = MNIST(train_data_dir)
    test_dataset = MNIST(test_data_dir)
    
    # Check length of dataset
    print("Length of dataset:", len(train_dataset))
    print("Length of dataset:", len(test_dataset))





