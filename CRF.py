import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from sklearn.metrics import confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

class EventExtractionCRF:
    def __init__(self):
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
    def load_data(self, file_path):
        """Load and parse the JSON data file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def preprocess_data(self, data):
        """Extract features and labels from the data"""
        X = []  # Features
        y = []  # Labels
        
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
                
                # Extract features for each token
                features = [self.extract_features(tokens, i) for i in range(len(tokens))]
                
                X.append(features)
                y.append(labels)
        
        return X, y
    
    def extract_features(self, tokens, i):
        """Extract features for a token at position i with improved context"""
        token = tokens[i] if i < len(tokens) else ''
        
        # Basic features
        features = {
            'bias': 1.0,
            'token': token.lower(),
            'token.is_digit': token.isdigit(),
            'token.is_title': token.istitle(),
            'token.is_lower': token.islower(),
            'token.is_upper': token.isupper(),
            'token.length': len(token),
        }
        
        # Character-level features
        if len(token) > 0:
            features['token.prefix1'] = token[0]
            features['token.suffix1'] = token[-1]
        if len(token) > 1:
            features['token.prefix2'] = token[:2]
            features['token.suffix2'] = token[-2:]
        if len(token) > 2:
            features['token.prefix3'] = token[:3]
            features['token.suffix3'] = token[-3:] if len(token) >= 3 else token
        
        # Context features - previous tokens
        for j in range(1, 3):  # Look at 2 previous tokens
            if i - j >= 0:
                prev_token = tokens[i-j].lower()
                features[f'prev{j}'] = prev_token
                features[f'prev{j}.is_title'] = tokens[i-j].istitle()
                # Combination features
                features[f'prev{j}+token'] = f"{prev_token}+{token.lower()}"
        
        # Context features - next tokens
        for j in range(1, 3):  # Look at 2 next tokens
            if i + j < len(tokens):
                next_token = tokens[i+j].lower()
                features[f'next{j}'] = next_token
                features[f'next{j}.is_title'] = tokens[i+j].istitle()
                # Combination features
                features[f'token+next{j}'] = f"{token.lower()}+{next_token}"
        
        # Position features
        features['position'] = i
        features['relative_position'] = float(i) / len(tokens) if len(tokens) > 0 else 0
        
        # Special cases
        if i == 0:
            features['BOS'] = True
        if i == len(tokens) - 1:
            features['EOS'] = True
        
        return features
    
    def train_with_epochs(self, X_train, y_train, X_dev, y_dev, num_epochs=10):
        """Train the CRF model with epoch tracking using progressive iterations"""
        # Store loss history (using F1 score as a proxy since CRF doesn't provide loss)
        f1_history = []
        
        # Create a new model for each epoch with progressive iterations
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Create a fresh CRF model for this epoch with more iterations
            epoch_model = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=(epoch+1) * 20,  # Increase iterations progressively
                all_possible_transitions=True
            )
            
            # Fit the model for this epoch
            epoch_model.fit(X_train, y_train)
            
            # Save the current model state
            self.model = epoch_model
            
            # Predict on dev set
            y_pred = self.model.predict(X_dev)
            
            # Calculate F1 score (higher is better, so we'll negate it to simulate loss)
            f1 = flat_f1_score(y_dev, y_pred, average='weighted')
            loss_proxy = 1.0 - f1  # Convert F1 to a loss-like metric (lower is better)
            f1_history.append(loss_proxy)
            
            print(f"Epoch {epoch+1} - F1 Score: {f1:.4f}, Loss Proxy: {loss_proxy:.4f}")
        
        # Train a final model with more iterations using all data
        print("Training final model with all data...")
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=200,  # More iterations for final model
            all_possible_transitions=True
        )
        self.model.fit(X_train, y_train)
        
        # Return the loss history for plotting
        return f1_history
    
    def train(self, X_train, y_train):
        """Train the CRF model (original method)"""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the CRF model"""
        y_pred = self.model.predict(X_test)
        report = flat_classification_report(y_test, y_pred)
        return report, y_pred
    
    def extract_event_types(self, labels_list):
        """Extract event types from labels for confusion matrix"""
        event_types = []
        valid_event_types = ["Leader-Activity", "Policy-Announcement", "Emergency-Event"]
        
        for labels in labels_list:
            event_type = None
            for label in labels:
                if label.startswith('B-TRIGGER-'):
                    event_type = label.replace('B-TRIGGER-', '')
                    # Ensure the extracted event type is valid
                    if event_type not in valid_event_types:
                        print(f"Warning: Found unexpected event type '{event_type}'")
                        event_type = None
                    break
            event_types.append(event_type if event_type else "None")
        
        return event_types

    def create_event_confusion_matrix(self, y_true, y_pred):
        """Create confusion matrix for event types"""
        # Extract event types
        true_events = self.extract_event_types(y_true)
        pred_events = self.extract_event_types(y_pred)
        
        # List of all event types (without None)
        event_types = ["Leader-Activity", "Policy-Announcement", "Emergency-Event"]
        
        # Filter out instances where either true or pred is None
        filtered_pairs = [(t, p) for t, p in zip(true_events, pred_events) 
                        if t != "None" and p != "None"]
        
        # If we have valid pairs after filtering
        if filtered_pairs:
            filtered_true, filtered_pred = zip(*filtered_pairs)
            
            try:
                # Create confusion matrix only for actual event types (no None)
                cm = confusion_matrix(filtered_true, filtered_pred, labels=event_types)
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=event_types, yticklabels=event_types)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix for Event Types')
                plt.tight_layout()
                plt.savefig('crf_confusion_matrix.png')
                print("Confusion matrix saved as 'crf_confusion_matrix.png'")
                plt.show()
            except Exception as e:
                print(f"Error creating confusion matrix: {str(e)}")
                print("True event counts:", Counter(filtered_true))
                print("Predicted event counts:", Counter(filtered_pred))
        else:
            print("No valid event pairs found after filtering out None values.")
            print("True event counts:", Counter(true_events))
            print("Predicted event counts:", Counter(pred_events))
    
    def plot_loss_history(self, loss_history):
        """Plot the loss history over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Proxy (1 - F1 Score)')
        plt.title('CRF Training Progress (Lower is Better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('crf_loss_history.png')
        print("Loss history plot saved as 'crf_loss_history.png'")
        plt.show()
    
    def save_model(self, file_path):
        """Save the trained model"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, file_path):
        """Load a trained model"""
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def get_sample_data(self, X, y, sample_size=100):
        """Extract a subset of the data for quick testing"""
        indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

if __name__ == "__main__":
    # Initialize model
    crf_model = EventExtractionCRF()
    
    # Load and preprocess data
    data = crf_model.load_data("tagged_events.json")
    X, y = crf_model.preprocess_data(data)
    
    print(f"Loaded {len(X)} samples")
    
    # Option to use a smaller subset for faster testing
    use_sample = False  # Set to True for faster testing with a subset
    if use_sample:
        X, y = crf_model.get_sample_data(X, y, sample_size=100)
        print(f"Using {len(X)} samples for testing")
    
    # Split data into train, dev, and test sets (70-15-15)
    # First split: 85% train+dev, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Second split: 70% train, 15% dev (which is 17.65% of the remaining 85%)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
    
    print(f"Training on {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Development on {len(X_dev)} samples ({len(X_dev)/len(X)*100:.1f}%)")
    print(f"Testing on {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model with epoch tracking (10 epochs)
    num_epochs = 50
    loss_history = crf_model.train_with_epochs(X_train, y_train, X_dev, y_dev, num_epochs=num_epochs)
    
    # Plot the loss history
    crf_model.plot_loss_history(loss_history)
    
    # Evaluate on test set
    test_report, y_pred = crf_model.evaluate(X_test, y_test)
    print("Test Set Performance Report:")
    print(test_report)
    
    # Create confusion matrix for event types
    crf_model.create_event_confusion_matrix(y_test, y_pred)
    
    # Save model
    crf_model.save_model("crf_event_extraction.pkl")
    print("Model saved to crf_event_extraction.pkl")