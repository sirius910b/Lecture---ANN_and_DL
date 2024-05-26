import torch
import torch.nn.functional as F
from model import CharRNN, CharLSTM

def generate(model, seed_characters, temperature, char2idx, idx2char, length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        char2idx: character to index mapping
        idx2char: index to character mapping
        length: number of characters to generate

    Returns:
        samples: generated characters
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):  # LSTM의 경우
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:  # RNN의 경우
        hidden = hidden.to(device)
    
    input_seq = torch.tensor([char2idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    
    samples = seed_characters
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output.squeeze() / temperature  # (batch_size * seq_length, vocab_size)
            probabilities = F.softmax(output, dim=-1)
            char_idx = torch.multinomial(probabilities, 1)[-1].item()  # select the last character in the sequence
            
            samples += idx2char[char_idx]
            input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return samples

if __name__ == '__main__':
    # 예제 코드: 모델 로드 및 샘플 생성
    import dataset
    input_file = 'shakespeare_train.txt'
    dataset_obj = dataset.Shakespeare(input_file)
    
    model_type = 'RNN'  # 'RNN' 또는 'LSTM' 중 하나 선택
    if model_type == 'RNN':
        model = CharRNN(len(dataset_obj.chars), hidden_size=256, num_layers=2)
        model.load_state_dict(torch.load('char_RNN.pth'))  # 훈련된 RNN 모델 가중치 로드
    elif model_type == 'LSTM':
        model = CharLSTM(len(dataset_obj.chars), hidden_size=256, num_layers=2)
        model.load_state_dict(torch.load('char_LSTM.pth'))  # 훈련된 LSTM 모델 가중치 로드
    else:
        raise ValueError("Invalid model type. Choose either 'RNN' or 'LSTM'.")

    seed_characters = "To be, or not to be, that is the question:"
    temperature = 0.8
    generated_text = generate(model, seed_characters, temperature, dataset_obj.char2idx, dataset_obj.idx2char)
    
    print(generated_text)



