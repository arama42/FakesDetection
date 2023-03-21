
!pip install transformers
!pip install datasets

# install libraries

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Adam
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import re
import nltk
import string


# Parse input data

def parse_bio(x, flag):
    if not x:
      return 0
    else:
      x = x.strip()
      index = x.rfind(" ")

      bio = x[:index]
      label_string = x[index+1:]

      if "FAKE" in label_string:
         label = 0
      elif "REAL" in label_string:
         label = 1
      else:
         label = 2
      
      if flag:
        return bio, label
      else:
        return bio

# Train
dirpath = "./data_cleaned/"
real_train = pd.read_csv(dirpath+"real.train.clean", names=["text"], delimiter="\"")
real_train["target"] = 1
fake_train = pd.read_csv(dirpath+"fake.train.clean", names=["text"], delimiter="\"")
fake_train["target"] = 0
train = pd.concat([real_train, fake_train], axis=0)
train = train.sample(frac=1).reset_index(drop=True)

# Validation
real_valid = pd.read_csv(dirpath+"real.valid.clean", names=["text"], delimiter="\"")
real_valid["target"] = 1
fake_valid = pd.read_csv(dirpath+"fake.valid.clean", names=["text"], delimiter="\"")
fake_valid["target"] = 0
valid = pd.concat([real_valid, fake_valid], axis=0)
valid = valid.sample(frac=1).reset_index(drop=True)

# Test
real_test = pd.read_csv(dirpath+"real.test.clean", names=["text"], delimiter="\"")
real_test["target"] = 1
fake_test = pd.read_csv(dirpath+"fake.test.clean", names=["text"], delimiter="\"")
fake_test["target"] = 0
test = pd.concat([real_test, fake_test], axis=0)
test = test.sample(frac=1).reset_index(drop=True)

# Mix
mix_train = pd.read_csv(dirpath+"mix.train.clean", names=["text"], delimiter="\"")
mix_train[['text', 'target']] = mix_train['text'].apply(lambda x: pd.Series(parse_bio(x,True)))

mix_valid = pd.read_csv(dirpath+"mix.valid.clean", names=["text"], delimiter="\"")
mix_valid[['text', 'target']] = mix_valid['text'].apply(lambda x: pd.Series(parse_bio(x, True)))

mix_test = pd.read_csv(dirpath+"mix.test.clean", names=["text"], delimiter="\"")
mix_test[['text', 'target']] = mix_test['text'].apply(lambda x: pd.Series(parse_bio(x, True)))

# Blind Test
blind_test = pd.read_csv(dirpath+"blind.test.clean", names=["text"], delimiter="\"")
blind_test[['text']] = blind_test['text'].apply(lambda x: pd.Series(parse_bio(x, False)))

"""### Preprocess data"""

class BioDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        
        texts = dataframe.text.values.tolist()
        #texts = [self._preprocess(text) for text in texts]

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        if 'target' in dataframe:
            classes = dataframe.target.values.tolist()
            self.labels = classes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label

"""### Model definition"""

class BioClassifier(nn.Module):
    def __init__(self, base_model):
        super(BioClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

"""###  Training loop definition"""

def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
    
        model.train()
        
        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            train_label = train_label.to(device)
            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label.float().unsqueeze(1))
            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0
            
            model.eval()
            
            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)
                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))
                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc
            
            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
            
            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val
                torch.save(model, f"best_model.pt")
                print("Saved the best model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1
                
            if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break

"""### Training Classifier"""

torch.manual_seed(0)
np.random.seed(0)
    
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
base_model = AutoModel.from_pretrained("roberta-base")

train_dataloader = DataLoader(BioDataset(mix_train, tokenizer), batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(BioDataset(mix_valid, tokenizer), batch_size=8, num_workers=0)

model = BioClassifier(base_model)

learning_rate = 1e-5
epochs = 5
train(model, train_dataloader, val_dataloader, learning_rate, epochs)

"""### Test on the test set"""

def get_text_predictions(model, loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    
    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)

            output = model(input_ids, attention_mask)
            output = (output > 0.5).int()
            results_predictions.append(output)
    
    return torch.cat(results_predictions).cpu().detach().numpy()

# get predicted labels
model = torch.load("best_model.pt")

test_dataloader = DataLoader(BioDataset(mix_test.drop(columns=['target']), tokenizer), batch_size=8, shuffle=False, num_workers=0)
mix_pred_targets = get_text_predictions(model, test_dataloader)

# Get F1 score

f1_mix = f1_score(mix_test["target"], mix_pred_targets, average='micro')
print("[Transformer Model] F1 score on mix test :", f1_mix)
acc_mix = accuracy_score(mix_test["target"], mix_pred_targets)
print("[Transformer Model] Accuracy score on mix test :", acc_mix)
disp = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(mix_test["target"], mix_pred_targets),display_labels=['Real','Fake'])
disp.plot()

"""### Test on Blind dataset"""

# Get predicted labels
model = torch.load("best_model.pt")

test_dataloader = DataLoader(BioDataset(blind_test, tokenizer), batch_size=8, shuffle=False, num_workers=0)
blind_pred_targets = get_text_predictions(model, test_dataloader)

# Save predicted labels
blind_pred_save = ['REAL' if pred.item()==1 else 'FAKE' for pred in blind_pred_targets]

blind_test['target'] = blind_pred_save
blind_test.to_csv('blind_test_predictions.csv', index=False)