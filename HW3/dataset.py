import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """Shakespeare dataset

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        self.chars = sorted(set(self.text))
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
        self.text_as_int = [self.char2idx[char] for char in self.text]
        self.seq_length = 30

    def __len__(self):
        return len(self.text_as_int) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        chunk = self.text_as_int[start_idx:end_idx]
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)
        return input_seq, target_seq

if __name__ == '__main__':
    dataset = Shakespeare(input_file='shakespeare_train.txt')
    print(f"Dataset length: {len(dataset)}")
    for i in range(3):
        input_seq, target_seq = dataset[i]
        print(f"Input sequence: {input_seq}")
        print(f"Target sequence: {target_seq}")




