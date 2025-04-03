import os
import click
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset


def execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        time_taken = end_time - start_time
        print(f'Function "{func.__name__}" executed in: {time_taken}')
        return result
    return wrapper


def rotate_batch_fixed_angle(images, angle=30):
    theta = torch.zeros(images.size(0), 2, 3)
    theta[:, 0, 0] = torch.cos(torch.deg2rad(torch.tensor(angle)))
    theta[:, 0, 1] = -torch.sin(torch.deg2rad(torch.tensor(angle)))
    theta[:, 1, 0] = torch.sin(torch.deg2rad(torch.tensor(angle)))
    theta[:, 1, 1] = torch.cos(torch.deg2rad(torch.tensor(angle)))
    
    grid = torch.nn.functional.affine_grid(theta, images.size(), align_corners=False)
    rotated_images = torch.nn.functional.grid_sample(images, grid, align_corners=False)
    return rotated_images


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)


class Classifier(nn.Module):
    def __init__(self, recurrent=False):
        super(Classifier, self).__init__()
        self.recurrent = recurrent
        print(f"Classifier mode: {'Recurrent' if self.recurrent else 'Linear'}")
        if recurrent:
            self.classifier = nn.LSTM(32, 32, num_layers=1, batch_first=True, proj_size=10)
        else:
            # self.classifier = nn.Linear(32, 10)
            self.classifier = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    
    def forward(self, x):
        if self.recurrent:
            out, (hn, cn) = self.classifier(x)
            out = out[:, -1, :]
        else:
            out = self.classifier(x)
        return out


class BaseTrainer():
    def __init__(self, *args, recurrent=False, **kwargs):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.recurrent = recurrent
        print(f"Current device use: {self.device}")
        self.set_dataloader()
        self.create_model()
        self.get_criterion_n_optimizer()
    
    def set_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    def create_model(self):
        if self.recurrent:
            self.encoder = Encoder().to(self.device)
            self.classifier = Classifier(recurrent=self.recurrent).to(self.device)
        else:
            self.model = nn.Sequential(Encoder(), Classifier()).to(self.device)
    
    def get_criterion_n_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        if self.recurrent:
            self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=0.001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def save(self):
        if not os.path.isdir('models'):
            os.makedirs("models")
        torch.save(self.model.state_dict(), "./models/base_trainer_model.pt")
    
    def load(self):
        self.model.load_state_dict(torch.load("./models/base_trainer_model.pt"))
    
    @execution_time
    def train(self, epochs=10):
        for epoch in range(epochs):
            total_loss = 0.0
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                if self.recurrent:
                    bs = data.size(0)
                    embedding_ori = self.encoder.forward(data)
                    data_left = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28)).view(bs, -1)
                    embedding_left = self.encoder.forward(data_left)
                    data_right = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28), angle=-30).view(bs, -1)
                    embedding_right = self.encoder.forward(data_right)

                    embedding = torch.stack([embedding_ori, embedding_left, embedding_right], dim=1)
                    outputs = self.classifier(embedding)
                else:
                    outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f} | avg_loss: {avg_loss:.4f}')

    @torch.no_grad()
    def evaluate(self):
        correct, total = 0, 0
        for data, labels in self.test_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            if self.recurrent:
                bs = data.size(0)
                embedding_ori = self.encoder.forward(data)
                data_left = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28)).view(bs, -1)
                embedding_left = self.encoder.forward(data_left)
                data_right = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28), angle=-30).view(bs, -1)
                embedding_right = self.encoder.forward(data_right)

                embedding = torch.stack([embedding_ori, embedding_left, embedding_right], dim=1)
                outputs = self.classifier(embedding)
            else:
                outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}% ({correct}/{total})')
        return accuracy


class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def create_model(self):
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.classifier = Classifier().to(self.device)
    
    def get_criterion_n_optimizer(self):
        self.autoencoder_criterion = nn.MSELoss()
        self.autoencoder_optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

        self.clf_criterion = nn.CrossEntropyLoss()
        self.clf_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
    
    @execution_time
    def train(self, epochs=10):
        print("AutoEncoder training...")
        for epoch in range(epochs):
            for data, _ in self.train_loader:
                data = data.to(self.device)
                embedding = self.encoder(data)
                output = self.decoder(embedding)
                loss = self.autoencoder_criterion(output, data)
                
                self.autoencoder_optimizer.zero_grad()
                loss.backward()
                self.autoencoder_optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("Classifier training...")
        for epoch in range(epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                with torch.no_grad():
                  embedding = self.encoder(data)
                outputs = self.classifier(embedding)
                loss = self.clf_criterion(outputs, labels)
                
                self.clf_optimizer.zero_grad()
                loss.backward()
                self.clf_optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    @torch.no_grad()
    def evaluate(self):
        correct, total = 0, 0
        for data, labels in self.test_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            embedding = self.encoder(data)
            outputs = self.classifier(embedding)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}% ({correct}/{total})')
        return accuracy


class PredictiveLinearNet():
    BASE_CONFIG = {'layers_conf': [(28*28, 128), (128, 64), (64, 32)], 'act_fn': nn.ReLU, 'first_act_fn': nn.Sigmoid,
                   'optimizer': optim.Adam, 'criterion': nn.MSELoss}
    def __init__(self, config={}):
        self.config = {**PredictiveLinearNet.BASE_CONFIG, **config}

        self.criterion = self.config['criterion']()
        self.optimizers = []

        layers, rec_layers = [], []
        for i, (n_in, n_out) in enumerate(self.config['layers_conf']):
            layers.append(nn.Sequential(nn.Linear(n_in, n_out), self.config['act_fn']()))
            rec_layers.append(nn.Sequential(nn.Linear(n_out, n_in), self.config['first_act_fn']() if i == 0 else self.config['act_fn']()))

            self.optimizers.append(self.config['optimizer'](list(layers[-1].parameters()) + list(rec_layers[-1].parameters())))

        self.layers = nn.ModuleList(layers)
        self.rec_layers = nn.ModuleList(rec_layers)
    
    def forward(self, x, layer_no=None):
        if layer_no:
            assert layer_no < len(self.layers)

        with torch.no_grad():
            for layer in self.layers[:layer_no]:  # if layer_no is None, it will go through all layers
                x = layer(x)

        if layer_no is None:  # inference
            return x
        
        out = self.layers[layer_no](x)
        rec_x = self.rec_layers[layer_no](out)
        return x, rec_x


class PredictiveTrainer(BaseTrainer):
    def __init__(self, *args, sum_views=False, recurrent=False, **kwargs):
        self.sum_views = sum_views
        print(f"Summing different views: {self.sum_views}")
        self.recurrent = recurrent
        super().__init__()
    
    def create_model(self):
        self.model = PredictiveLinearNet()
        self.classifier = Classifier(recurrent=self.recurrent)
    
    def get_criterion_n_optimizer(self):
        self.clf_criterion = nn.CrossEntropyLoss()
        self.clf_trainer = optim.Adam(self.classifier.parameters())
    
    def train_model_layer(self, data, layer_no):
        out, rec = self.model.forward(data, layer_no=layer_no)
        loss = self.model.criterion(rec, out)
        self.model.optimizers[layer_no].zero_grad()
        loss.backward()
        self.model.optimizers[layer_no].step()
        return loss
    
    @execution_time
    def train(self, epochs = 7):
        for i in range(len(self.model.layers)):
            print(f"Train layer {i+1}...")
            for epoch in range(epochs):
                for data, _ in self.train_loader:
                    data = data.to(self.device)
                    loss = self.train_model_layer(data, i)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("Train classifier...")
        for epoch in range(epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    if self.sum_views or self.recurrent:
                        bs = data.size(0)
                        embedding_ori = self.model.forward(data)
                        data_left = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28)).view(bs, -1)
                        embedding_left = self.model.forward(data_left)
                        data_right = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28), angle=-30).view(bs, -1)
                        embedding_right = self.model.forward(data_right)

                        if self.sum_views:
                            embedding = embedding_ori + embedding_left + embedding_right
                        else:
                            embedding = torch.stack([embedding_ori, embedding_left, embedding_right], dim=1)
                    else:
                        embedding = self.model.forward(data)
                out = self.classifier(embedding)
                loss = self.clf_criterion(out, labels)
                self.clf_trainer.zero_grad()
                loss.backward()
                self.clf_trainer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    @torch.no_grad()
    def evaluate(self):
        correct, total = 0, 0
        for data, labels in self.test_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            if self.sum_views or self.recurrent:
                bs = data.size(0)
                embedding_ori = self.model.forward(data)
                data_left = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28)).view(bs, -1)
                embedding_left = self.model.forward(data_left)
                data_right = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28), angle=-30).view(bs, -1)
                embedding_right = self.model.forward(data_right)

                if self.sum_views:
                    embedding = embedding_ori + embedding_left + embedding_right
                else:
                    embedding = torch.stack([embedding_ori, embedding_left, embedding_right], dim=1)
            else:
                embedding = self.model.forward(data)
            outputs = self.classifier(embedding)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}% ({correct}/{total})')
        return accuracy


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    

class MNISTTripletDataset(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        self.dataset = torchvision.datasets.MNIST(root='./data', train=train, transform=transform, download=True)
        self.data_by_label = {i: [] for i in range(10)}
        for idx, (image, label) in enumerate(self.dataset):
            self.data_by_label[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]
        
        # Select positive sample (same class)
        pos_idx = np.random.choice(self.data_by_label[anchor_label])
        positive_img, _ = self.dataset[pos_idx]
        
        # Select negative sample (different class)
        negative_label = np.random.choice([l for l in range(10) if l != anchor_label])
        neg_idx = np.random.choice(self.data_by_label[negative_label])
        negative_img, _ = self.dataset[neg_idx]

        return anchor_img.view(-1), positive_img.view(-1), negative_img.view(-1), anchor_label
    

class ContrastiveTrainer():
    def __init__(self, *args, **kwargs):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(f"Current device use: {self.device}")
        self.set_dataloader()
        self.create_model()
        self.get_criterion_n_optimizer()

    def set_dataloader(self):
        train_dataset = MNISTTripletDataset(train=True)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

        test_dataset = MNISTTripletDataset(train=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    def create_model(self):
        self.encoder = Encoder().to(self.device)
        self.classifier = Classifier().to(self.device)

    def get_criterion_n_optimizer(self):
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.encoder_criterion = TripletLoss()

        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.classifier_criterion = nn.CrossEntropyLoss()
    
    @execution_time
    def train(self, n_epochs=10):
        print(f"Training Encoder...")
        self.encoder.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for anchor, positive, negative, _ in self.train_loader:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                anchor_embed = self.encoder(anchor)
                positive_embed = self.encoder(positive)
                negative_embed = self.encoder(negative)

                loss = self.encoder_criterion(anchor_embed, positive_embed, negative_embed)
                self.encoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()

                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss / len(self.train_loader):.4f}")
        
        print("Training Classifier...")
        self.classifier.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for images, _, _, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    embeddings = self.encoder(images)
                outputs = self.classifier(embeddings)
                loss = self.classifier_criterion(outputs, labels)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss / len(self.train_loader):.4f}")

    @torch.no_grad()
    def evaluate(self):
        self.encoder.eval()
        self.classifier.eval()
        correct, total = 0, 0
        for images, _, _, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            embeddings = self.encoder(images)
            outputs = self.classifier(embeddings)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}% ({correct}/{total})')
        return accuracy


@click.command()
@click.option("--trainer", "-t", "trainer", type=str, default="base")
@click.option("--sum-views", "-s", "sum_views", is_flag=True)
@click.option("--recurrent", "-r", is_flag=True)
@click.option("--eval-all", "-e", 'eval_all', is_flag=True)
def main(trainer, sum_views, recurrent, eval_all):
    trainers = {'base': BaseTrainer, 'autoencoder': AutoEncoderTrainer, 'predictive': PredictiveTrainer,
                'contrastive': ContrastiveTrainer}

    if eval_all:
        to_show = {'trainer': [], 'sum_views': [], 'recurrent': [], 'accuracy': []}
        for k, v in trainers.items():
            for sum_views in [False, True]:
                for recurrent  in [False, True]:
                    if (sum_views and recurrent) or ((sum_views or recurrent) and k != 'predictive'):
                        continue
                    torch.manual_seed(42)
                    ctrainer = v(sum_views=sum_views, recurrent=recurrent)
                    ctrainer.train()
                    cacc = ctrainer.evaluate()
                    to_show['trainer'].append(k)
                    to_show['sum_views'].append(sum_views)
                    to_show['recurrent'].append(recurrent)
                    to_show['accuracy'].append(f"{cacc:.2f}")
        print(tabulate(pd.DataFrame.from_dict(to_show), headers='keys', tablefmt='psql'))
    else:
        torch.manual_seed(42)
        # torch.set_float32_matmul_precision("high")
        ctrainer = trainers[trainer](sum_views=sum_views, recurrent=recurrent)
        ctrainer.train()
        # import code; code.interact(local=locals())
        ctrainer.evaluate()


if __name__ == "__main__":
    main()
    # RESULTS: 
    # No hyperparameters tuning done
    #
    # BaseTrainer = Fully-connected encoding + classification layer
    # Accuracy = 97.15% | done in 20s
    #
    # AutoEncoderTrainer = Fully-connected Encoder-Decoder training then classification layer training
    # Accuracy = 90.17% | done in 43s
    #
    # PredictiveTrainer base = Train each fully-connected layers independently using reconstruction loss
    # Accuracy = 87.12% | done in 48s
    #
    # PredictiveTrainer sum-views = sum 3 views of the same image before trying to classified it
    # Accuracy = 86.63% | done in 58s
    #
    # PredictiveTrainer rnn-classifier = encode 3 views of the same image then provide them to a recurrent network
    # Accuracy = 95.45% | done in 1mn03s
    #
    # PredictiveTrainer with bigger linear classifier
    # To test if the gain obtained with the RNN classifier is just because it's expressive power if bigger
    # let's increase the simple classifier from Linear() to Sequential(Linear, ReLU, Linear)
    # Accuracy = 93.07% | done in 47s
    # PS: increasing the hidden size of the classifier doesn't help
    #     e.g. from 64 to 128 give lower accuracy of 91.93% done in 59s
    #
    # PredictiveTrainer with bigger linear classifier and concatenated views
    # The code is no longer here but an experiment was conduct with concatenated views instead of sum
    # and a bigger classifier (Sequential(Linear(3*32, 32), ReLU, Linear))
    # Accuracy = 92.50% | done in 59s