import torch
import torch.nn as nn
import torch.optim as optim

# ====================================
# 1. Training Sentences
# ====================================

sentences = [
    "i love ai",
    "ai is amazing",
    "deep learning is powerful",
    "python is awesome",
    "machine learning is fun",
    "i enjoy deep learning",
    "ai helps people",
    "deep learning uses neural networks"
]

# ====================================
# 2. Tokenization
# ====================================

tokenized_sentences = [s.lower().split() for s in sentences]

# ====================================
# 3. Build Vocabulary
# ====================================

vocab = {}
index = 0
for sentence in tokenized_sentences:
    for word in sentence:
        if word not in vocab:
            vocab[word] = index
            index += 1

# Reverse mapping
index_to_word = {idx: word for word, idx in vocab.items()}

print("Vocabulary:", vocab)

# ====================================
# 4. Create Training Pairs
# ====================================

X_data, y_data = [], []
for sentence in tokenized_sentences:
    for i in range(1, len(sentence)):
        input_words = sentence[:i]
        target_word = sentence[i]
        X_data.append(input_words)
        y_data.append(vocab[target_word])

# ====================================
# 5. Convert Words → Indices
# ====================================

X_sequences = [[vocab[word] for word in seq] for seq in X_data]

# ====================================
# 6. Padding
# ====================================

max_length = max(len(seq) for seq in X_sequences)
padded_sequences = [seq + [0]*(max_length - len(seq)) for seq in X_sequences]

# ====================================
# 7. Tensors
# ====================================

X = torch.tensor(padded_sequences, dtype=torch.long)
y = torch.tensor(y_data, dtype=torch.long)

print("\nInput Shape:", X.shape)
print("Target Shape:", y.shape)

# ====================================
# 8. LSTM Model
# ====================================

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        final_hidden = hidden[-1]
        out = self.fc(final_hidden)
        return out

# ====================================
# 9. Initialize Model
# ====================================

vocab_size = len(vocab)
model = NextWordLSTM(vocab_size=vocab_size, embedding_dim=16, hidden_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ====================================
# 10. Training
# ====================================

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ====================================
# 11. Prediction Function
# ====================================

def predict_next_word(text):
    words = text.lower().split()
    sequence = [vocab[word] for word in words if word in vocab]
    sequence += [0] * (max_length - len(sequence))

    tensor = torch.tensor([sequence], dtype=torch.long)

    with torch.no_grad():
        output = model(tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    predicted_word = index_to_word[predicted_index]
    print(f"\nInput: {text}")
    print(f"Predicted Next Word: {predicted_word}")

# ====================================
# 12. Test Predictions
# ====================================

predict_next_word("i love")
predict_next_word("ai is")
predict_next_word("deep learning")
predict_next_word("python is")
