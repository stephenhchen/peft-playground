from torch.utils.data import Dataset
import pandas as pd

class CommonSenseDataset(Dataset):
    def __init__(self, fpath, tokenizer, max_length=512):
        self.df = pd.read_json(fpath)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        instruction = row['instruction']
        answer = row['answer']
        
        # Combine instruction and answer with a separator
        text = f"{instruction}\n{answer}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create the labels (same as input_ids for causal language modeling)
        labels = encodings['input_ids'].clone()
        
        # Convert to PyTorch tensors
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        labels = labels.squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }