import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import time
from pathlib import Path
from pandas import DataFrame
from random import shuffle

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
PRETRAINED_MODEL = "bert-base-uncased"
BATCH_SIZE = 32
MAX_LENGTH = 128
OUTPUT = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25

# Below are several different number of datasets that will be used for training.
DATASET_1 = "thePixel42/depression-detection"
DATASET_1_TEST = True
DATASET_2 = "joangaes/depression"
DATASET_2_TEST = False
DATASET_3 = "mangoesai/DepressionDetection"
DATASET_3_TEST = True
DATASET_4 = "ziq/depression_tweet"
DATASET_4_TEST = True


def create_and_display_dataframe(dictionary):
    # This function is used to create a dataframe of the model results using PANDASSSSSSSSSSSSSSSSS!!!!
    dataframe = DataFrame(dictionary)
    print(dataframe)
    return dataframe


def set_data_device(device, X, y):
    # Sets the data's device and datatype

    X = X.to(device).type(torch.long)
    y = y.to(device).type(torch.float)
    return X, y


class DatasetManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # The lists below save the entered datasets allowing for easy data manipulation.
        self.dataset_storage_train = []
        self.dataset_storage_test = []
        self.tokenized_dataset_train = []
        self.tokenized_dataset_test = []
        self.train_dict = {}
        self.test_dict = {}


    def load_huggingface_data(self, dataset, contains_test_files):
        # This function loads the dataset from huggingface.

        if contains_test_files:
            train_dataset = load_dataset(dataset, split="train")
            test_dataset = load_dataset(dataset, split="test")
        else:
            # splits the train file into both train and test data.

            train_test_split= load_dataset(dataset, split="train").train_test_split(test_size=.2)
            train_dataset = train_test_split["train"]
            test_dataset = train_test_split["test"]

        self.dataset_storage_train.append(train_dataset)
        self.dataset_storage_test.append(test_dataset)
        
        return train_dataset, test_dataset

    def tokenize(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    def bugfix_tokenize(self, example):
        return self.tokenizer(example["clean_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    def tokenize_dataset(self, train_dataset, test_dataset):
        # This function tokenizes the inputted datasets and converts them to torch.
        try:
            tokenized_train_dataset = train_dataset.map(self.tokenize, batched=True, batch_size=BATCH_SIZE)
            tokenized_test_dataset = test_dataset.map(self.tokenize, batched=True, batch_size=BATCH_SIZE)
        except Exception:
            tokenized_train_dataset = train_dataset.map(self.bugfix_tokenize, batched=True, batch_size=BATCH_SIZE)
            tokenized_test_dataset = test_dataset.map(self.bugfix_tokenize, batched=True, batch_size=BATCH_SIZE)

        try:
            tokenized_train_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"]
            )
            tokenized_test_dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"]
            )
        
        except Exception:
            tokenized_train_dataset.set_format(
                type="torch",
                columns=["clean_text", "is_depression", "input_ids", "token_type_ids", "attention_mask"]
            )
            tokenized_test_dataset.set_format(
                type="torch",
                columns=["clean_text", "is_depression", "input_ids", "token_type_ids", "attention_mask"]
            )

        self.tokenized_dataset_train.append(tokenized_train_dataset)
        self.tokenized_dataset_test.append(tokenized_test_dataset)

        return tokenized_train_dataset, tokenized_test_dataset
    
    def tokenize_all_saved_datasets(self):
        # This function tokenizes all the saved datasets. This makes it easier to tokenize all the data.

        for (train, test) in zip(self.dataset_storage_train, self.dataset_storage_test):
            self.tokenize_dataset(train, test)

    def merge_tokenized_datasets(self):
        # This function merges all of the datasets into a dict containing the X(input) and y(target).
        # Test and Training will not be merged because we need to keep test seperate from train so we can evaluate the Transformer model.

        # Initial values that will be used to store the inputs(X) and outputs(y).
        test_X = []
        test_y = []
        train_X = []
        train_y = []

        for train_dataset in self.tokenized_dataset_train:
            train_X += train_dataset["input_ids"]  # input
            try:
                train_y += train_dataset["label"]  # output
            except Exception:
                train_y += train_dataset["is_depression"]  # also output

        for test_dataset in self.tokenized_dataset_test:
            test_X += test_dataset["input_ids"]  # input
            try:
                test_y += test_dataset["label"]  # output
            except Exception:
                test_y += test_dataset["is_depression"]  # also output


        train_dataset = {"X": train_X, "y": train_y}
        test_dataset = {"X": test_X, "y": test_y}

        self.train_dict = train_dataset
        self.test_dict = test_dataset

        return train_dataset, test_dataset
    
    def shuffle_data(self, dataset):
        # As the title suggests this function shuffles the data

        merged_data = list(zip(dataset["X"], dataset["y"]))
        shuffle(merged_data)
        dataset["X"], dataset["y"] = zip(*merged_data)
        return dataset
    
    def apply_shuffle(self):
        # This function applies the shuffle_data function to the test and train dict.

        self.train_dict = self.shuffle_data(self.train_dict)
        self.test_dict = self.shuffle_data(self.test_dict)

    def batch_data(self, dataset=None):
        # This function batches the data in order to train a neural network.

        batch = 1
        X_batch = []
        y_batch = []
        X_list = []
        y_list = []
        for i, (text, category) in enumerate(zip(dataset["X"], dataset["y"])):
            X_batch.append(text)
            y_batch.append(category)
            if i+1 == BATCH_SIZE*batch or len(dataset["X"]) == i+1:
                batch += 1
                X = torch.stack(X_batch, dim=0)
                y = torch.stack(y_batch, dim=0)
                X_batch.clear()
                y_batch.clear()
                X_list.append(X)
                y_list.append(y)
        dataset = {"X": X_list, "y": y_list}
        return dataset
    
    def apply_batch(self):
        self.train_dict = self.batch_data(self.train_dict)
        self.test_dict = self.batch_data(self.test_dict)

    def prepare_all_loaded_data(self):
        print("Preparing data...")
        self.tokenize_all_saved_datasets()
        self.merge_tokenized_datasets()
        self.apply_shuffle()
        self.apply_batch()
        print("Data sucessfully prepared")
        return self.train_dict, self.test_dict
    

class ClassifierModel(nn.Module):
    def __init__(self, classes, pretrained_model):
        super(ClassifierModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, classes)

    def forward(self, x, attention_mask):
        x = self.pretrained_model(input_ids=x, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x
    

def create_attention_mask(X):
    # This function creates an attention mask for the transformer model.
    # This mask is used to mask padding tokens (0s) in the input.

    attention_mask = (X != 0)
    attention_mask = attention_mask.to(DEVICE)
    return attention_mask


def calculate_accuracy(y_logits, y):
    y_pred = torch.sigmoid(y_logits)
    y_pred = (y_pred > .5).int()
    accuracy = (y_pred == y).sum().item()
    accuracy = accuracy/y.size(0)
    return accuracy * 100


def save_model(model):
    # function the automatically saves the model

    model = model.to(DEVICE)

    print("Saving model...")
    model_path = Path("Depression_detecting_models")
    model_path.mkdir(parents=True, exist_ok=True)
    model_name = "Model_002_pretraind.pth"
    model_save_path = model_path/model_name
    torch.save(model, model_save_path)
    model_state_dict_name = "Model_002_pretrained_state_dict.pth"
    dict_path = model_path/model_state_dict_name
    torch.save(model.state_dict, dict_path)
    print("Model has been saved!")


def train(dataloader, model, loss_fn, optimizer):
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    model.train()
    for batch, (X, y) in enumerate(zip(dataloader["X"], dataloader["y"])):
        X, y = set_data_device(DEVICE, X, y)

        attention_mask = create_attention_mask(X)

        y_logits = model.forward(X, attention_mask=attention_mask)
        loss = loss_fn(y_logits.squeeze(dim=1), y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
        train_accuracy += accuracy
    train_loss = train_loss/len(dataloader["X"])
    train_accuracy = train_accuracy/len(dataloader["X"])
    end = time.time()
    train_time = end-start
    return train_loss, train_accuracy, train_time


def test(dataloader, model, loss_fn):
    start = time.time()
    test_loss = 0
    test_accuracy = 0
    model.eval()

    with torch.no_grad():
        for batch, (X, y) in enumerate(zip(dataloader["X"], dataloader["y"])):
            X, y = set_data_device(DEVICE, X, y)

            attention_mask = create_attention_mask(X)

            y_logits = model.forward(X, attention_mask=attention_mask)
            loss = loss_fn(y_logits.squeeze(dim=1), y)
            # print(f"loss: {loss}")
            # print(f"y_logits: {y_logits}")
            test_loss += loss
            accuracy = calculate_accuracy(y_logits.squeeze(dim=1), y)
            test_accuracy += accuracy
        test_loss = test_loss/len(dataloader["X"])
        test_accuracy = test_accuracy/len(dataloader["X"])
        end = time.time()
        test_time = end-start
        return test_loss, test_accuracy, test_time


def main():
    dataset_manager = DatasetManager(TOKENIZER)
    dataset_manager.load_huggingface_data(DATASET_1, DATASET_1_TEST)
    dataset_manager.load_huggingface_data(DATASET_2, DATASET_2_TEST)
    dataset_manager.load_huggingface_data(DATASET_3, DATASET_3_TEST)
    dataset_manager.load_huggingface_data(DATASET_4, DATASET_4_TEST)

    train_dataloader, test_dataloader = dataset_manager.prepare_all_loaded_data()

    model = ClassifierModel(OUTPUT, AutoModel.from_pretrained(PRETRAINED_MODEL))
    model = model.to(DEVICE)

    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    results = {"Epoch": [], "Train loss": [], "Train accuracy": [], "Train time": [], "Test loss": [],
               "Test Accuracy": [], "Test time": []}
    print("Starting training process...")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy, train_time = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_accuracy, test_time = test(test_dataloader, model, loss_fn)
        print(f"\nEpoch: {epoch+1}\nTrain loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.3f}% | "
              f"Train time: {train_time:.2f}\nTest loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.3f}% | "
              f"Test time: {test_time:.2f}")
        results["Epoch"].append(epoch+1)
        results["Train loss"].append(round(train_loss.detach().item(), 4))
        results["Train accuracy"].append(round(train_accuracy, 3))
        results["Train time"].append(round(train_time, 2))
        results["Test loss"].append(round(test_loss.detach().item(), 4))
        results["Test Accuracy"].append(round(test_accuracy, 3))
        results["Test time"].append(round(test_time, 2))
    create_and_display_dataframe(results)
    save_model(model)


if __name__ == "__main__":
    main()
