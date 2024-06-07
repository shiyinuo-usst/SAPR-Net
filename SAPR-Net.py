import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from scipy.io import savemat

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(21)

# Self-Attention mechanism used in Transformer
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        device = self.values.weight.device
        values, keys, queries = values.to(device), keys.to(device), queries.to(device)
        if mask is not None:
            mask = mask.to(device)
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Linear transformation and splitting into multiple heads
        values = self.values(values.reshape(N, value_len, self.heads, self.head_dim))
        keys = self.keys(keys.reshape(N, key_len, self.heads, self.head_dim))
        queries = self.queries(queries.reshape(N, query_len, self.heads, self.head_dim))

        # Scaled dot-product attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(attention / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)

# Cross-Attention mechanism
class CrossAttention(nn.Module):
    def __init__(self, input_size):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads=1)

    def forward(self, input1, input2):
        cross_attention_output, _ = self.attention(input1, input2, input2)
        return cross_attention_output

# Normalize positional encoding
def normalize_positional_encoding(positional_encoding):
    mean = positional_encoding.mean()
    std = positional_encoding.std()
    return (positional_encoding - mean) / std

# Simple Retention mechanism
class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, head_size=None, double_v_dim=False):
        super(SimpleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size if head_size else hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = 1

        self.W_Q1 = nn.Parameter(torch.randn(7, head_size))
        self.W_K1 = nn.Parameter(torch.randn(7, head_size))
        self.W_V1 = nn.Parameter(torch.randn(7, head_size))

        self.W_Q2 = nn.Parameter(torch.randn(7, head_size))
        self.W_K2 = nn.Parameter(torch.randn(7, head_size))
        self.W_V2 = nn.Parameter(torch.randn(7, head_size))

        self.W_Q3 = nn.Parameter(torch.randn(7, head_size))
        self.W_K3 = nn.Parameter(torch.randn(7, head_size))
        self.W_V3 = nn.Parameter(torch.randn(7, head_size))

        self.W_Q4 = nn.Parameter(torch.randn(7, head_size))
        self.W_K4 = nn.Parameter(torch.randn(7, head_size))
        self.W_V4 = nn.Parameter(torch.randn(7, head_size))

    def forward(self, X):
        batch_size, _, hidden_size = X.shape

        # Split input into four chunks
        chunk1 = X[:, :7, :7]
        chunk2 = X[:, :7, 7:]
        chunk3 = X[:, 7:, :7]
        chunk4 = X[:, 7:, 7:]

        r_i_1 = torch.zeros(batch_size, 7, 7).to(X.device)

        # Process each chunk
        output1, r_i = self.process_chunk(chunk1, r_i_1, 0)
        output2, r_i = self.process_chunk(chunk2, r_i, 1)
        output3, r_i = self.process_chunk(chunk3, r_i, 2)
        output4, r_i = self.process_chunk(chunk4, r_i, 3)

        # Concatenate the outputs
        upper_half = torch.cat([output1, output2], dim=2)
        lower_half = torch.cat([output3, output4], dim=2)
        return torch.cat([upper_half, lower_half], dim=1)

    def process_chunk(self, x_i, r_i_1, chunk_index):
        W_Q, W_K, W_V = [self.W_Q1, self.W_K1, self.W_V1,
                         self.W_Q2, self.W_K2, self.W_V2,
                         self.W_Q3, self.W_K3, self.W_V3,
                         self.W_Q4, self.W_K4, self.W_V4][chunk_index*3:chunk_index*3+3]

        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_index).to(self.device)

        Q = x_i @ W_Q
        K = x_i @ W_K
        V = x_i @ W_V
        r_i = K.transpose(-1, -2) @ V + r_i_1

        inner_chunk = (Q @ K.transpose(-1, -2) * D.unsqueeze(0)) @ V
        e = torch.zeros(batch, chunk_size, 1, device=x_i.device)
        for _i in range(chunk_size):
            e[:, _i, :] = 0.2 ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e
        return inner_chunk + cross_chunk, r_i

    def _get_D(self, chunk_index):
        df_D = pd.read_csv('C:/Users/shiyinuo/PycharmProjects/brain recognition/attenuation_matrix.csv', header=None)
        D_full = torch.tensor(df_D.values).float()
        if chunk_index == 0:
            D = D_full[:7, :7]
        elif chunk_index == 1:
            D = D_full[:7, 7:]
        elif chunk_index == 2:
            D = D_full[7:, :7]
        else:
            D = D_full[7:, 7:]
        return D.to(self.device)

# Transformer block consisting of self-attention and feed-forward layers
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.retention = SimpleRetention(embed_size)
        self.norm1 = nn.LayerNorm(14)
        self.norm2 = nn.LayerNorm(14)
        self.feed_forward = nn.Sequential(
            nn.Linear(14, forward_expansion * 14),
            nn.ReLU(),
            nn.Linear(forward_expansion * 14, 14),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.retention(x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))

# Multi-Input Transformer model
class MultiInputTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, forward_expansion, num_channels, num_classes, dropout, pos_encoding_path):
        super(MultiInputTransformer, self).__init__()
        self.embed_size = embed_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_channels = num_channels
        self.additional_conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(7, 7), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.cross_attention1 = CrossAttention(14)
        self.cross_attention2 = CrossAttention(14)
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_size, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]) for _ in range(num_channels)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = self.load_and_normalize_pos_encoding(pos_encoding_path)
        self.fc_out = nn.Linear(980, num_classes)

    def load_and_normalize_pos_encoding(self, file_path):
        pos_encoding_df = pd.read_csv(file_path, header=None)
        pos_encoding = pos_encoding_df.values
        return torch.tensor(normalize_positional_encoding(pos_encoding), dtype=torch.float32)

    def forward(self, x):
        out = []
        split_input = torch.split(x, 1, dim=1)
        for input_per_band, transformer_blocks in zip(split_input, self.transformer_blocks):
            transformed = self.transform_to_matrix(input_per_band)
            transformed += self.pos_encoding.to(transformed.device)
            for block in transformer_blocks:
                transformed = block(transformed)
            out.append(transformed)
        out_1 = torch.stack(out, dim=1)
        cross_output_1 = self.cross_attention1(out[0], out[2])
        cross_output_2 = self.cross_attention2(out[1], out[3])
        final_cross_output = self.cross_attention1(cross_output_1, cross_output_2).unsqueeze(1)
        combined_output = torch.cat([out_1, final_cross_output], dim=1).view(x.size(0), -1)
        return self.fc_out(combined_output)

    def transform_to_matrix(self, x):
        x_reshaped = x.view(x.shape[0], 14, 1)
        return torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))

# Load and process data from CSV file
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    data = df.values[:, :-1].reshape(-1, 4, 14)
    labels = df.values[:, -1]
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Custom dataset class for EEG data
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Training loop for one epoch
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct_predictions += torch.sum(torch.max(outputs, 1)[1] == labels)
    return running_loss / len(data_loader.dataset), correct_predictions.double() / len(data_loader.dataset)

# Testing loop with confusion matrix calculation
def test_epoch_with_confusion_matrix(model, data_loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    cm = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.max(outputs, 1)[1]
            correct_predictions += torch.sum(preds == labels)
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                all_preds.append(pred)
                all_labels.append(label)
                cm[label][pred] += 1
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss / len(data_loader.dataset), correct_predictions.double() / len(data_loader.dataset), cm, f1

# Save confusion matrix as .mat file
def save_confusion_matrix_as_mat(cm, filename="confusion_matrix.mat"):
    savemat(filename, {'confusion_matrix': cm})

# Load training and testing data
train_data, train_labels = load_and_process_data('C:/Users/shiyinuo/PycharmProjects/brain recognition/30/1_2345_train.csv')
test_data, test_labels = load_and_process_data('C:/Users/shiyinuo/PycharmProjects/brain recognition/30/1_2345_test.csv')

train_dataset = EEGDataset(train_data, train_labels)
test_dataset = EEGDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Set device and model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORWARD_EXPANSION = 40
DROPOUT = 0.5
NUM_LAYERS = 1
EMBED_SIZE = 7
NUM_CHANNELS = 4
NUM_CLASSES = 30

# Initialize model, criterion, and optimizer
model = MultiInputTransformer(
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    forward_expansion=FORWARD_EXPANSION,
    num_channels=NUM_CHANNELS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
    pos_encoding_path='C:/Users/shiyinuo/PycharmProjects/brain recognition/2D_positional encoding.csv'
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize confusion matrix with accuracy annotations
def visualize_confusion_matrix_with_accuracy(cm):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=False, cmap='Blues')
    for i in range(cm.shape[0]):
        total_predictions = np.sum(cm[i])
        accuracy = (cm[i, i] / total_predictions) * 100 if total_predictions != 0 else 0
        ax.text(i + 0.5, i + 0.5, f"{accuracy:.0f}%", horizontalalignment='center', verticalalignment='center', color='white', fontsize=8)
    plt.title('Phase One', fontsize=20)
    plt.show()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}")

# Test and visualize results
test_loss, test_acc, cm, f1 = test_epoch_with_confusion_matrix(model, test_loader, criterion, device, NUM_CLASSES)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, F1 Score: {f1}")
print(f"Confusion Matrix Shape: {cm.shape}")

visualize_confusion_matrix_with_accuracy(cm)
