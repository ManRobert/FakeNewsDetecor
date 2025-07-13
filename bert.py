import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df['statement'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def load_datasets(tokenizer, max_length=128):
    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("valid.csv")
    df_test = pd.read_csv("test.csv")

    train_dataset = FakeNewsDataset(df_train, tokenizer, max_length)
    val_dataset = FakeNewsDataset(df_val, tokenizer, max_length)
    test_dataset = FakeNewsDataset(df_test, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def train_model(train_loader, val_loader, model, optimizer, device, epochs=5, patience=2):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch + 1} train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "bert_fakenews_multiclass_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device, name="Evaluation"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    report = classification_report(labels, preds, digits=4)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"conf_matrix_{name}.png")
    plt.close()


def classify_single_news(text, tokenizer, model, device, max_length=128):
    model.eval()
    encoding = tokenizer(text,
                         truncation=True,
                         padding='max_length',
                         max_length=max_length,
                         return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets(tokenizer)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=6
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("Select option:")
    print("1 - Train and evaluate model")
    print("2 - Input a news statement and classify")

    while True:
        option = input("Enter option number: ").strip()

        if option == "1":
            train_model(train_loader, val_loader, model, optimizer, device, epochs=5, patience=2)
            print("\nEvaluating model on test set...")
            model.load_state_dict(torch.load("bert_fakenews_multiclass_best.pt"))
            evaluate_model(model, test_loader, device, name="After_FineTuning")

        elif option == "2":
            model.load_state_dict(torch.load("bert_fakenews_multiclass_best.pt"))
            text = input("Enter the news statement to classify:\n")
            pred = classify_single_news(text, tokenizer, model, device)
            print(f"Predicted class: {pred}")

        else:
            print("Invalid option. Exiting.")

if __name__ == "__main__":
    main()
