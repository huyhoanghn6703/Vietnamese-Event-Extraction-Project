import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=100, hidden_dim=200):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        # Embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters, entry i,j is the score of transitioning from j to i
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        
        # These two statements enforce constraints that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[:, self.tag_to_ix["START"]] = -10000
        self.transitions.data[self.tag_to_ix["STOP"], :] = -10000
        
    def _forward_alg(self, feats):
        # Calculate the partition function in log-space using the forward algorithm
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score
        init_alphas[0][self.tag_to_ix["START"]] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["STOP"]]
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha
    
    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix["START"]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix["STOP"], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix["START"]] = 0
        
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning from tag i to next_tag
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
            # Add the emission scores
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix["STOP"]]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        # Follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        

        best_path.pop()  # Pop off the start tag
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats[0])
        gold_score = self._score_sentence(feats[0], tags)
        return forward_score - gold_score
    
    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        
        # Find the best path using Viterbi
        score, tag_seq = self._viterbi_decode(lstm_feats[0])
        return score, tag_seq

class EventExtractionDataset(Dataset):
    def __init__(self, X, y, word_to_ix, tag_to_ix):
        self.X = X
        self.y = y
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = [self.word_to_ix.get(w.lower(), self.word_to_ix["<UNK>"]) for w in self.X[idx]]
        y = [self.tag_to_ix[t] for t in self.y[idx]]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def load_and_prepare_data(file_path):
    """Load and prepare data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    tags = []
    
    for doc in data:
        if "tokens" not in doc or "event_mentions" not in doc or not doc["event_mentions"]:
            continue
            
        tokens = doc["tokens"]
        if not tokens:
            continue
            
        for event in doc["event_mentions"]:
            if "trigger" not in event or "arguments" not in event:
                continue
                
            # Initialize labels as 'O' (Outside) for all tokens
            labels = ['O'] * len(tokens)
            
            # Mark trigger
            if "start" in event["trigger"] and "end" in event["trigger"]:
                start, end = event["trigger"]["start"], event["trigger"]["end"]
                if 0 <= start < end <= len(tokens):
                    if end - start == 1:
                        labels[start] = f'B-TRIGGER-{event["event_type"]}'
                    else:
                        labels[start] = f'B-TRIGGER-{event["event_type"]}'
                        for i in range(start + 1, end):
                            labels[i] = f'I-TRIGGER-{event["event_type"]}'
            
            # Mark arguments
            for arg in event["arguments"]:
                if "role" in arg and "start" in arg and "end" in arg:
                    start, end = arg["start"], arg["end"]
                    if 0 <= start < end <= len(tokens):
                        if end - start == 1:
                            labels[start] = f'B-{arg["role"]}'
                        else:
                            labels[start] = f'B-{arg["role"]}'
                            for i in range(start + 1, end):
                                labels[i] = f'I-{arg["role"]}'
            
            sentences.append(tokens)
            tags.append(labels)
    
    return sentences, tags

def create_vocabularies(sentences, tags):
    """Create word and tag vocabularies"""
    word_to_ix = {"<PAD>": 0, "<UNK>": 1, "START": 2, "STOP": 3}
    for sentence in sentences:
        for word in sentence:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    # Add special tags START and STOP
    tag_to_ix = {"START": 0, "STOP": 1, "O": 2}
    for sentence_tags in tags:
        for tag in sentence_tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    return word_to_ix, tag_to_ix

def train_model(model, train_loader, optimizer, num_epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train the BiLSTM-CRF model"""
    model.train()
    
    # Store loss history for plotting
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for sentence, tags in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move to device
            sentence = sentence.to(device)
            tags = tags.to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            loss = model.neg_log_likelihood(sentence, tags[0])
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loss_history.png')
    plt.show()
    
    return loss_history

def evaluate_model(model, test_loader, tag_to_ix, ix_to_tag, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluate the BiLSTM-CRF model"""
    model.eval()
    
    y_true = []
    y_pred = []
    
    # For confusion matrix of event types
    event_true = []
    event_pred = []
    event_types = ["Leader-Activity", "Policy-Announcement", "Emergency-Event"]
    
    with torch.no_grad():
        for sentence, tags in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            sentence = sentence.to(device)
            
            # Get model predictions
            _, predicted_tags = model(sentence)
            
            # Convert indices to tag names
            true_tags = [ix_to_tag[tag.item()] for tag in tags[0]]
            pred_tags = [ix_to_tag[tag.item()] if isinstance(tag, torch.Tensor) else ix_to_tag[tag] for tag in predicted_tags]
            
            # Extract event types for confusion matrix
            true_event_type = None
            pred_event_type = None
            
            for tag in true_tags:
                if tag.startswith("B-TRIGGER-"):
                    true_event_type = tag.replace("B-TRIGGER-", "")
                    break
            
            for tag in pred_tags:
                if tag.startswith("B-TRIGGER-"):
                    pred_event_type = tag.replace("B-TRIGGER-", "")
                    break
            
            if true_event_type is not None:
                event_true.append(true_event_type)
                event_pred.append(pred_event_type if pred_event_type is not None else "None")
            
            y_true.append(true_tags)
            y_pred.append(pred_tags)
    
    # Print classification report
    print(classification_report(y_true, y_pred))
    
    # Create confusion matrix for event types
    # Filter out None event types
    valid_events = [(t, p) for t, p in zip(event_true, event_pred) if p != "None"]
    if valid_events:
        # Unzip the pairs
        event_true_filtered, event_pred_filtered = zip(*valid_events)
        
        # Create confusion matrix
        cm = confusion_matrix(event_true_filtered, event_pred_filtered, labels=event_types)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=event_types, yticklabels=event_types)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for Event Types')
        plt.tight_layout()
        plt.savefig('event_confusion_matrix.png')
        plt.show()
    else:
        print("No valid event predictions found for confusion matrix.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    sentences, tags = load_and_prepare_data("tagged_events.json")
    print(f"Loaded {len(sentences)} sentences with tags")
    
    # Create vocabularies
    word_to_ix, tag_to_ix = create_vocabularies(sentences, tags)
    ix_to_tag = {i: t for t, i in tag_to_ix.items()}
    print(f"Vocabulary size: {len(word_to_ix)}")
    print(f"Number of tags: {len(tag_to_ix)}")
    
    # Split data into train, dev, and test sets (70-15-15)
    # First split: 85% train+dev, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(sentences, tags, test_size=0.15, random_state=42)
    
    # Second split: 70% train, 15% dev (which is 17.65% of the remaining 85%)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
    
    print(f"Training on {len(X_train)} samples ({len(X_train)/len(sentences)*100:.1f}%)")
    print(f"Development on {len(X_dev)} samples ({len(X_dev)/len(sentences)*100:.1f}%)")
    print(f"Testing on {len(X_test)} samples ({len(X_test)/len(sentences)*100:.1f}%)")
    
    # Create datasets and dataloaders
    train_dataset = EventExtractionDataset(X_train, y_train, word_to_ix, tag_to_ix)
    dev_dataset = EventExtractionDataset(X_dev, y_dev, word_to_ix, tag_to_ix)
    test_dataset = EventExtractionDataset(X_test, y_test, word_to_ix, tag_to_ix)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize model
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix).to(device)
    
    # Training parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 50
    
    # Train model and get loss history
    loss_history = train_model(model, train_loader, optimizer, num_epochs, device)
    
    # Evaluate on development set
    print("Development Set Performance:")
    evaluate_model(model, dev_loader, tag_to_ix, ix_to_tag, device)
    
    # Evaluate on test set
    print("Test Set Performance:")
    evaluate_model(model, test_loader, tag_to_ix, ix_to_tag, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_ix': word_to_ix,
        'tag_to_ix': tag_to_ix,
        'loss_history': loss_history
    }, "bilstm_crf_event_extraction.pt")
    print("Model saved to bilstm_crf_event_extraction.pt")