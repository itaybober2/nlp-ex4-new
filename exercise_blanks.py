import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import data_loader
from data_loader import get_negated_polarity_examples, get_rare_words_examples
import pickle

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    embeddings = []
    unknown_count = 0
    for word_node in sent.get_leaves():
        word = word_node.text[0]
        if word in word_to_vec:
            embeddings.append(word_to_vec[word])
        else:
            unknown_count += 1

    total_words = len(sent.get_leaves())
    print(f"Unknown words: {unknown_count}/{total_words} ({unknown_count / total_words:.2%})")

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    one_hot_vectors = []
    for word_node in sent.get_leaves():
        word = word_node.text[0]  # Extract the word from the node
        if word in word_to_ind:  # Only include words in the vocabulary
            one_hot_vector = get_one_hot(len(word_to_ind), word_to_ind[word])
            one_hot_vectors.append(one_hot_vector)

    # Compute the average one-hot embedding
    if one_hot_vectors:  # Avoid division by zero for empty sentences
        avg_one_hot = np.mean(one_hot_vectors, axis=0)
    else:
        avg_one_hot = np.zeros(len(word_to_ind))  # Return a zero vector for empty sentences
    return avg_one_hot


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: idx for idx, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    """
    Map the given sentence to its word embeddings, with padding or truncation to seq_len.
    """

    embeddings = []
    for word_node in sent.get_leaves():
        word = word_node.text[0]
        if word in word_to_vec:
            embeddings.append(word_to_vec[word])
        else:
            embeddings.append(np.zeros(embedding_dim))  # Unknown words are zero vectors

    # Pad or truncate to the required sequence length
    if len(embeddings) > seq_len:
        embeddings = embeddings[:seq_len]
    else:
        embeddings.extend([np.zeros(embedding_dim)] * (seq_len - len(embeddings)))

    return np.array(embeddings)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the Bi-directional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )

        # Define a dropout layer
        self.dropout = nn.Dropout(dropout)

        # Define the final linear layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Bi-directional, hence *2

    def forward(self, text):
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(text)

        # Take the final hidden states of both directions
        forward_hidden = lstm_out[:, -1, :self.hidden_dim]
        backward_hidden = lstm_out[:, 0, self.hidden_dim:]

        # Concatenate forward and backward hidden states
        final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        # Apply dropout and the fully connected layer
        output = self.fc(self.dropout(final_hidden))
        return output

    def predict(self, text):
        # Predict probabilities
        with torch.no_grad():
            logits = self.forward(text)
            return torch.sigmoid(logits)


class LogLinear(nn.Module):
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        return self.sigmoid(logits)


    def predict(self, x):
        """
        Perform predictions on the input.
        :param x: Input tensor
        :return: Binary predictions (0 or 1)
        """
        with torch.no_grad():
            probabilities = self.forward(x)
            return (probabilities >= 0.5).float()  # Convert probabilities to binary predictions


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    binary_preds = (preds >= 0.5).float()
    correct = (binary_preds == y).float().sum()
    accuracy = correct / y.size(0)
    return accuracy.item()


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    model.train()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for x_batch, y_batch in data_iterator:
        x_batch = x_batch.float()
        y_batch = y_batch.float()

        optimizer.zero_grad()
        predictions = torch.sigmoid(model(x_batch).squeeze(1))
        loss = criterion(model(x_batch).squeeze(1), y_batch)
        acc = binary_accuracy(predictions, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        num_batches += 1

    return epoch_loss / num_batches, epoch_acc / num_batches


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in data_iterator:
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            predictions = model(x_batch).squeeze(1)
            loss = criterion(predictions, y_batch)
            acc = binary_accuracy(predictions, y_batch)

            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1

    return epoch_loss / num_batches, epoch_acc / num_batches


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch, _ in data_iter:
            x_batch = x_batch.float()
            preds = model(x_batch).squeeze(1)
            predictions.extend(preds.cpu().numpy())

    return np.array(predictions)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    return


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # Step 1: Initialize DataManager
    data_manager = DataManager(data_type="onehot_average")  # Use one-hot average embeddings
    train_iterator = data_manager.get_torch_iterator("train")
    val_iterator = data_manager.get_torch_iterator("val")

    # Step 2: Initialize the model, optimizer, and loss function
    vocab_size = len(data_manager.sentiment_dataset.get_word_counts())
    model = LogLinear(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

    # Step 3: Train the model
    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_iterator, criterion)

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Step 4: Evaluate the model on the test set
    test_iterator = data_manager.get_torch_iterator("test")
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def train_log_linear_with_w2v():
    """
    Train and evaluate the log-linear model using Word2Vec embeddings.
    """
    # Step 1: Initialize DataManager
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=300)
    train_iterator = data_manager.get_torch_iterator(TRAIN)
    val_iterator = data_manager.get_torch_iterator(VAL)
    test_iterator = data_manager.get_torch_iterator(TEST)

    # Step 2: Initialize the model, optimizer, and loss function
    embedding_dim = 300
    model = LogLinear(embedding_dim)
    model.to(get_available_device())

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.BCELoss()

    n_epochs = 20

    # Step 3: Training loop
    for epoch in range(n_epochs):
        # Train the model for one epoch
        train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion)

        # Evaluate the model on the validation set
        val_loss, val_acc = evaluate(model, val_iterator, criterion)

        # Log the results for this epoch
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print("-" * 50)

    # Step 4: Evaluate on the test set
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print("Final Test Results")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Step 5: Evaluate on special subsets
    special_subsets = {
        "Negated Polarity": data_manager.sentiment_dataset.get_test_set(),
        "Rare Words": data_manager.sentiment_dataset.get_test_set()
    }

    for subset_name, subset_sentences in special_subsets.items():
        # Obtain indices for special subsets
        if subset_name == "Negated Polarity":
            subset_indices = get_negated_polarity_examples(subset_sentences)
        elif subset_name == "Rare Words":
            subset_indices = get_rare_words_examples(subset_sentences, data_manager.sentiment_dataset)

        # Filter data iterator for the special subset
        subset_data = [subset_sentences[i] for i in subset_indices]
        subset_dataset = OnlineDataset(subset_data, data_manager.sent_func, data_manager.sent_func_kwargs)
        subset_iterator = DataLoader(subset_dataset, batch_size=64, shuffle=False)

        # Evaluate the model on the subset
        subset_loss, subset_acc = evaluate(model, subset_iterator, criterion)
        print(f"{subset_name} Results:")
        print(f"Loss: {subset_loss:.4f}, Accuracy: {subset_acc:.4f}")
        print("-" * 50)



def train_lstm_with_w2v():
    # Initialize DataManager
    data_manager = DataManager(data_type="w2v_sequence", batch_size=64, embedding_dim=300)
    train_iterator = data_manager.get_torch_iterator("train")
    val_iterator = data_manager.get_torch_iterator("val")

    # Define model, optimizer, and loss function
    device = get_available_device()
    model = LSTM(embedding_dim=300, hidden_dim=100, n_layers=1, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss().to(device)  # Combines sigmoid and binary cross-entropy

    # Train the model for 4 epochs
    n_epochs = 4
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_iterator, criterion)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print("-" * 50)

    # Test the model on the test set
    test_iterator = data_manager.get_torch_iterator("test")
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print("Final Test Results")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Evaluate on special subsets
    print("\nEvaluating on special subsets...")
    special_subsets = {
        "Negated Polarity": get_negated_polarity_examples(data_manager.sentiment_dataset.get_test_set()),
        "Rare Words": get_rare_words_examples(data_manager.sentiment_dataset.get_test_set(), data_manager.sentiment_dataset)
    }

    for subset_name, subset_indices in special_subsets.items():
        subset_data = [data_manager.sentiment_dataset.get_test_set()[i] for i in subset_indices]
        subset_dataset = OnlineDataset(subset_data, data_manager.sent_func, data_manager.sent_func_kwargs)
        subset_iterator = DataLoader(subset_dataset, batch_size=64, shuffle=False)

        subset_loss, subset_acc = evaluate(model, subset_iterator, criterion)
        print(f"{subset_name} Results:")
        print(f"Loss: {subset_loss:.4f}, Accuracy: {subset_acc:.4f}")
        print("-" * 50)


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()