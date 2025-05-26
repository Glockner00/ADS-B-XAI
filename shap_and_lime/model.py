import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

THRESHOLD = 0.7

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.n_features))
        _, (hidden_n, _) = self.rnn1(x)
        _, (hidden_n, _) = self.rnn2(hidden_n)
        return hidden_n.reshape((-1, self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# WRAPPER FOR SHAP 
class LossThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold, device="cpu", seq_len=8, n_features=1):
        self.model = model
        self.threshold = threshold
        self.device = device
        self.seq_len = seq_len
        self.n_features = n_features

    def predict(self, X):
        losses = self._get_losses(X)
        return np.array([0 if l <= self.threshold else 1 for l in losses])

    def predict_proba(self, X):
        feature_losses = self._get_feature_losses(X)
        return feature_losses

    def _get_losses(self, X):
        losses = []
        for x in X:
            x_tensor = torch.tensor(
                x.reshape(self.seq_len, self.n_features), dtype=torch.float32
            ).to(self.device)
            _, loss = self._predict([x_tensor])
            losses.append(loss[0])
        return losses

    def _get_feature_losses(self, X):
        feature_losses = []
        for x in X:
            x_tensor = torch.tensor(
                x.reshape(self.seq_len, self.n_features), dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            with torch.no_grad():
                x_hat = self.model(x_tensor.unsqueeze(0))
            loss_per_feature = torch.mean(torch.abs(x_tensor - x_hat.squeeze(0)), dim=0)
            feature_losses.append(loss_per_feature.cpu().numpy())
        return np.array(feature_losses)

    def _predict(self, X):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction="sum").to(self.device)
        self.model.eval()
        with torch.no_grad():
            for x in X:
                seq_true = x.to(self.device)
                seq_pred = self.model(seq_true)
                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
        return predictions, losses