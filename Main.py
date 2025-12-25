"""
Turkish Phishing Email Detection Project
PyTorch Implementation with Transformer Models + TF-IDF Baseline
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
from tqdm import tqdm
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EmailData:
    """Structure to hold email information"""
    subject: str
    body: str
    sender: str
    headers: Dict[str, str]
    urls: List[str]
    label: int  # 0: legitimate, 1: phishing


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoaderUtil:
    """Handles loading and splitting phishing email datasets"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from email body text"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # Comprehensive URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        # Also look for www. patterns without http
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        www_urls = re.findall(www_pattern, text)
        
        return list(set(urls + www_urls))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 1. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # 2. Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # 3. Remove standalone single digits
        text = re.sub(r'\b\d\b', '', text)
        
        return text.strip()
    
    def load_english_dataset(self) -> List[EmailData]:
        """Load dataset"""
        print(f"Loading dataset from: {self.data_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(self.data_path, encoding=encoding)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any standard encoding")
            
            initial_count = len(df)
            
            # Drop rows where body is NaN
            df = df.dropna(subset=['body'])
            
            # Removes rows where Subject AND Body are identical.
            df = df.drop_duplicates(subset=['subject', 'body'], keep='first')
            
            dropped_count = initial_count - len(df)
            if dropped_count > 0:
                print(f"Dropped {dropped_count} duplicate/empty emails to prevent data leakage.")

            print(f"Loaded {len(df)} unique emails")
            print(f"Label distribution:\n{df['label'].value_counts()}")
            
            # Convert to EmailData objects
            emails = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing emails"):
                # Extract URLs from body
                body_text = str(row['body']) if pd.notna(row['body']) else ""
                urls = self.extract_urls_from_text(body_text)
                
                # Clean text fields
                subject = self.clean_text(str(row['subject']) if pd.notna(row['subject']) else "")
                body = self.clean_text(body_text)
                sender = self.clean_text(str(row['sender']) if pd.notna(row['sender']) else "")
                
                # Create headers dict
                headers = {
                    'date': str(row['date']) if pd.notna(row['date']) else "",
                    'receiver': str(row['receiver']) if pd.notna(row['receiver']) else ""
                }
                
                # Get label (ensure it's 0 or 1)
                label = int(row['label'])
                
                email = EmailData(
                    subject=subject,
                    body=body,
                    sender=sender,
                    headers=headers,
                    urls=urls,
                    label=label
                )
                emails.append(email)
            
            print(f"Successfully processed {len(emails)} emails")
            
            return emails
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            raise
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
    
    def split_data(self, emails: List[EmailData], test_size: float = 0.15, 
                   val_size: float = 0.15, random_state: int = 42) -> Tuple[List, List, List]:
        """Split data into train, validation, and test sets with stratification"""
        labels = [e.label for e in emails]
        
        # First split: separate test set
        train_val, test = train_test_split(
            emails, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: separate train and validation
        train_val_labels = [e.label for e in train_val]
        train, val = train_test_split(
            train_val, 
            test_size=val_size/(1-test_size), 
            random_state=random_state,
            stratify=train_val_labels
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(train)} emails")
        print(f"  Val:   {len(val)} emails")
        print(f"  Test:  {len(test)} emails")
        
        return train, val, test


# ============================================================================
# MACHINE TRANSLATION
# ============================================================================

class MachineTranslator:
    """Batched translation from English to Turkish using Marian NMT"""

    def __init__(
        self,
        model_path: str = "models/opus-mt-tc-big-en-tr",
        device: str = None,
        batch_size: int = 32,
        max_length: int = 128
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_path = model_path

        print(f"Loading translation model from: {model_path}")

        model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
                
        if not os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                force_download=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                force_download=True,
                use_safetensors=True
            )

            os.makedirs(model_path, exist_ok=True)
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path)

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True,
                use_safetensors=True
            )

        self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()

        print(f"Translator ready on {self.device}")

    def _batch_translate_texts(self, texts: list[str], desc: str = "Translating") -> list[str]:
        results = []

        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            total=num_batches,
            desc=desc
        ):
            batch = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=1,
                    do_sample=False,
                )

            decoded = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            results.extend(decoded)

        return results

    def _protect_urls(self, text: str, urls: list[str]):
        placeholders = {}
        protected = text

        for i, url in enumerate(urls):
            token = f"__URL_{i}__"
            placeholders[token] = url
            protected = protected.replace(url, token)

        return protected, placeholders

    def _restore_urls(self, text: str, placeholders: dict):
        restored = text
        for token, url in placeholders.items():
            restored = restored.replace(token, url)
        return restored

    def batch_translate_dataset(
        self,
        emails: list[EmailData],
        save_path: str | None = None
    ) -> list[EmailData]:

        if save_path and os.path.exists(save_path):
            print(f"Loading cached translations from {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        print("Preparing texts for batched translation...")

        subjects = [e.subject for e in emails]

        protected_bodies = []
        body_placeholders = []

        for e in emails:
            text, placeholders = self._protect_urls(e.body, e.urls)
            protected_bodies.append(text)
            body_placeholders.append(placeholders)

        translated_subjects = self._batch_translate_texts(
            subjects,
            desc="Translating subjects"
        )

        translated_bodies = self._batch_translate_texts(
            protected_bodies,
            desc="Translating bodies"
        )

        translated_emails = []
        for i, e in enumerate(emails):
            body = self._restore_urls(
                translated_bodies[i], body_placeholders[i]
            )

            translated_emails.append(
                EmailData(
                    subject=translated_subjects[i],
                    body=body,
                    sender=e.sender,
                    headers=e.headers,
                    urls=e.urls,
                    label=e.label
                )
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(translated_emails, f)
            print(f"Saved translations to {save_path}")

        return translated_emails


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Preprocess text for model input"""
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
    
    def preprocess(self, email: EmailData) -> str:
        """Preprocess email text"""
        text = f"{email.subject} [SEP] {email.body}"
        
        if self.lowercase:
            text = text.lower()
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# ============================================================================
# TF-IDF BASELINE CLASSIFIER
# ============================================================================

class TFIDFClassifier:
    """Traditional ML baseline using TF-IDF + Logistic Regression"""
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2), min_df: int = 2, max_df: float = 0.95, 
                C: float = 1.0, max_iter: int = 1000, sublinear_tf: bool = True, solver: str = 'liblinear'):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf
        )
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
            random_state=42,
            class_weight='balanced'
        )
        self.preprocessor = TextPreprocessor()
    
    def train(self, train_emails: List[EmailData], val_emails: List[EmailData] = None):
        """Train TF-IDF + LR model"""
        print("Extracting TF-IDF features from training data...")
        
        # Preprocess texts
        train_texts = [self.preprocessor.preprocess(e) for e in train_emails]
        train_labels = [e.label for e in train_emails]
        
        # Fit vectorizer and transform
        X_train = self.vectorizer.fit_transform(train_texts)
        
        print(f"Training logistic regression on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.classifier.fit(X_train, train_labels)
        
        # Validation accuracy if provided
        if val_emails:
            val_texts = [self.preprocessor.preprocess(e) for e in val_emails]
            val_labels = [e.label for e in val_emails]
            X_val = self.vectorizer.transform(val_texts)
            val_acc = self.classifier.score(X_val, val_labels)
            print(f"Validation accuracy: {val_acc:.4f}")
    
    def evaluate(self, test_emails: List[EmailData]) -> Dict[str, float]:
        """Evaluate on test set"""
        print("Evaluating TF-IDF model...")
        
        test_texts = [self.preprocessor.preprocess(e) for e in test_emails]
        test_labels = [e.label for e in test_emails]
        
        X_test = self.vectorizer.transform(test_texts)
        predictions = self.classifier.predict(X_test)
        probabilities = self.classifier.predict_proba(X_test)[:, 1]
        
        report = classification_report(test_labels, predictions, output_dict=True)
        cm = confusion_matrix(test_labels, predictions)
        
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions))
        print(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_roc': roc_auc_score(test_labels, probabilities),
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, path: str):
        """Save vectorizer and classifier"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
        print(f"TF-IDF model saved to {path}")
    
    def load_model(self, path: str):
        """Load vectorizer and classifier"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
        print(f"TF-IDF model loaded from {path}")


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class EmailDataset(Dataset):
    """PyTorch Dataset for email data"""
    
    def __init__(self, emails: List[EmailData], tokenizer, preprocessor, max_length: int = 256):
        self.emails = emails
        self.tokenizer = tokenizer
        self.encodings = tokenizer(
            [preprocessor.preprocess(e) for e in emails],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email = self.emails[idx]
        
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(email.label, dtype=torch.long)
        }


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerClassifier(nn.Module):
    """Transformer-based classifier for phishing detection"""
    
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.3, freeze_layers: int = 0):
        super().__init__()
        self.model_name = model_name
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Freezing logic
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        if freeze_layers > 0:
            encoder_layers = self.transformer.encoder.layer
            freeze_layers = min(freeze_layers, len(encoder_layers))
            for layer in encoder_layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Classifier using transformer hidden size
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = self.dropout(transformer_out.last_hidden_state[:, 0, :])
        
        return self.classifier(pooled)


# ============================================================================
# PHISHING DETECTOR
# ============================================================================

class PhishingDetector:
    """Main detector class for training and evaluation"""
    
    def __init__(self, model_name: str, max_length: int = 256, learning_rate: float = 2e-5,
                 batch_size: int = 32, epochs: int = 3, patience: int = 2, freeze_layers: int = 0, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TransformerClassifier(model_name, freeze_layers=freeze_layers).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = TextPreprocessor()
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
    
    def create_dataloader(self, emails: List[EmailData], shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader"""
        dataset = EmailDataset(emails, self.tokenizer, self.preprocessor, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self, train_emails: List[EmailData], val_emails: List[EmailData]):
        """Train the model"""
        train_loader = self.create_dataloader(train_emails, shuffle=True)
        val_loader = self.create_dataloader(val_emails, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        weights = torch.tensor([1.0, 1.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        return history
    
    def _validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return val_loss / len(val_loader), correct / total
    
    def evaluate(self, test_emails: List[EmailData]) -> Dict[str, float]:
        """Evaluate model on test set"""
        test_loader = self.create_dataloader(test_emails, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report = classification_report(all_labels, all_predictions, output_dict=True)
        cm = confusion_matrix(all_labels, all_predictions)
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        print(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_roc': roc_auc_score(all_labels, all_probabilities),
            'confusion_matrix': cm.tolist()
        }
    
    def predict(self, email: EmailData) -> Tuple[int, float]:
        """Predict single email"""
        self.model.eval()
        text = self.preprocessor.preprocess(email)
        
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return prediction.item(), probabilities[0, prediction].item()
    
    def save_model(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded from {path}")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
    
    def run_tfidf(self,
                      exp_name: str,
                      train_data: List[EmailData],
                      val_data: List[EmailData],
                      test_data: List[EmailData],
                      hyperparams: Dict = None):
        """Run TF-IDF + Logistic Regression"""
        print(f"\nRunning Baseline: {exp_name}")
        print("-" * 50)
        print(f"Method: TF-IDF + Logistic Regression")
        print(f"Training Size: {len(train_data)}")
        
        # Default params if not provided
        params = hyperparams or {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'C': 1.0,
            'max_iter': 1000,
        }
        
        classifier = TFIDFClassifier(**params)
        classifier.train(train_data, val_data)
        metrics = classifier.evaluate(test_data)
        
        self.results[exp_name] = {
            "model": "TF-IDF + Logistic Regression",
            "train_size": len(train_data),
            "hyperparams": params,
            "metrics": metrics
        }
        
        save_name = exp_name.replace(" ", "_").lower()
        classifier.save_model(f"{self.output_dir}/{save_name}.pkl")
    
    def run_experiment(self,
                       exp_name: str,
                       model_name: str,
                       train_data: List[EmailData],
                       val_data: List[EmailData],
                       test_data: List[EmailData],
                       freeze_layers: int = 0,
                       hyperparams: Dict = None):
        
        print(f"\nRunning Experiment: {exp_name}")
        print("-" * 50)
        print(f"Model: {model_name}")
        print(f"Training Size: {len(train_data)}")
        
        params = hyperparams or {'learning_rate': 2e-5, 'batch_size': 32, 'epochs': 3, 'patience': 2}
        
        detector = PhishingDetector(
            model_name=model_name, 
            freeze_layers=freeze_layers,
            **params
        )
        
        detector.train(train_data, val_data)
        metrics = detector.evaluate(test_data)
        
        self.results[exp_name] = {
            "model": model_name,
            "train_size": len(train_data),
            "metrics": metrics
        }
        
        save_name = exp_name.replace(" ", "_").lower()
        detector.save_model(f"{self.output_dir}/{save_name}.pt")

    def run_all_experiments(self, english_emails, translated_emails):
        
        tr_train, tr_val, tr_test = DataLoaderUtil("").split_data(translated_emails)
        
        en_train, en_val, _ = DataLoaderUtil("").split_data(english_emails)
        
        # TF-IDF Baseline on Turkish Data
        self.run_tfidf(
            exp_name="TFIDF_Turkish_Baseline",
            train_data=tr_train,
            val_data=tr_val,
            test_data=tr_test
        )
        
        # BERTurk
        self.run_experiment(
            exp_name="BERTurk",
            model_name="dbmdz/bert-base-turkish-cased",
            train_data=tr_train, val_data=tr_val, test_data=tr_test,
            freeze_layers=4
        )

        # XLM-R Zero-Shot
        self.run_experiment(
            exp_name="XLMR_ZeroShot",
            model_name="xlm-roberta-base",
            train_data=en_train, val_data=en_val, test_data=tr_test,
            freeze_layers=8
        )
        
        # Create dataset: 50 TR samples
        def sample_balanced(emails, n):
            phishing = [e for e in emails if e.label == 1]
            legit = [e for e in emails if e.label == 0]
            k = n // 2
            return random.sample(phishing, k) + random.sample(legit, k)
        
        few_shot_50 = sample_balanced(tr_train, 50)
        random.shuffle(few_shot_50)
        
        # TF-IDF Few-Shot (50 TR)
        self.run_tfidf(
            exp_name="TFIDF_FewShot_50TR",
            train_data=few_shot_50,
            val_data=tr_val[:300],
            test_data=tr_test,
            hyperparams={
                'max_features': 3000,
                'ngram_range': (1, 2),
                'min_df': 1,
                'max_df': 0.95,
                'sublinear_tf': True,
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'liblinear'
            }
        )
        
        # XLM-R Few-Shot (50 TR)
        self.run_experiment(
            exp_name="XLMR_FewShot_50TR",
            model_name="xlm-roberta-base",
            train_data=few_shot_50, val_data=tr_val[:300], test_data=tr_test,
            freeze_layers=8,
            hyperparams={'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 5, 'patience': 3}
        )
        
        # Create dataset: 100 TR + 200 EN
        mixed_train = tr_train[:100] + en_train[:200]
        random.shuffle(mixed_train)
        
        # Mixed Data XLM-R
        self.run_experiment(
            exp_name="XLMR_Mixed_100TR_200EN",
            model_name="xlm-roberta-base",
            train_data=mixed_train, val_data=tr_val, test_data=tr_test,
            freeze_layers=8,
            hyperparams={'learning_rate': 2e-5, 'batch_size': 32, 'epochs': 5, 'patience': 2}
        )

        # Compare Results
        return self.compare_results()

    def compare_results(self):
        """Print and save comparison of all approaches"""
        print("\n" + "="*80)
        print("FINAL RESULTS COMPARISON")
        print("="*80 + "\n")

        # Flatten metrics
        rows = []
        for name, result in self.results.items():
            metrics = result.get("metrics", {})
            row = {
                "approach": name,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "auc_roc": metrics.get("auc_roc"),
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("approach")

        print(df.round(4))
        print("\n")

        # Save results
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_csv(f"{self.output_dir}/comparison_results.csv")

        json_path = f"{self.output_dir}/results_{self.run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {json_path}")

        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("="*80)
    print("Turkish Phishing Detection Project")
    print("="*80)
    
    # Hardware check
    print(f"PyTorch version: {torch.__version__}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("="*80 + "\n")

    # Configuration
    DATASET_PATH = "datasets/ByNaser/CEAS_08.csv"
    TRANSLATION_CACHE = "./results/translated_emails.pkl"

    # Load English dataset
    print("Step 1: Loading source dataset...")
    loader = DataLoaderUtil(DATASET_PATH)
    try:
        english_emails = loader.load_english_dataset()
    except Exception as e:
        print(f"Critical Error: Failed to load dataset. {e}")
        return

    # Prepare translated dataset
    print("\nStep 2: Preparing translated dataset...")

    if os.path.exists(TRANSLATION_CACHE):
        print(f"Found cached translations at {TRANSLATION_CACHE}")
        with open(TRANSLATION_CACHE, "rb") as f:
            translated_emails = pickle.load(f)
        print(f"Loaded {len(translated_emails)} translated emails")
    else:
        print("No cached translations found — running machine translation")
        # Initialize translator only if needed to save VRAM
        translator = MachineTranslator(
            batch_size=32,
            max_length=128
        )

        translated_emails = translator.batch_translate_dataset(
            english_emails,
            save_path=TRANSLATION_CACHE
        )
        
        # Free up VRAM after translation is done
        del translator
        torch.cuda.empty_cache()

    # Run Experiment
    print("\n" + "="*80)
    print("STARTING EXPERIMENTS")
    print("="*80)
    
    # Initialize the runner
    runner = ExperimentRunner(output_dir="./results")

    # Run the full ablation study
    runner.run_all_experiments(english_emails, translated_emails)

    return


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    main()