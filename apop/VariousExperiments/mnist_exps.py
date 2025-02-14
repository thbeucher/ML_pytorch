import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datetime import datetime
from torch.utils.data import DataLoader


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
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.model = nn.Sequential(Encoder(), Classifier()).to(self.device)
    
    def get_criterion_n_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
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
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}%')


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
        print(f'Classifier Accuracy: {accuracy:.2f}%')


class PredictiveTrainer(BaseTrainer):
    def __init__(self, *args, sum_views=False, recurrent=False, **kwargs):
        self.sum_views = sum_views
        print(f"Summing different views: {self.sum_views}")
        self.recurrent = recurrent
        super().__init__()
    
    def create_model(self):
        self.fc1 = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

        self.rec_fc1 = nn.Sequential(nn.Linear(128, 28*28), nn.Sigmoid())
        self.rec_fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.rec_fc3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())

        self.classifier = Classifier(recurrent=self.recurrent)
    
    def get_criterion_n_optimizer(self):
        self.predictive_criterion = nn.MSELoss()
        self.clf_criterion = nn.CrossEntropyLoss()

        self.fc1_trainer = optim.Adam(list(self.fc1.parameters()) + list(self.rec_fc1.parameters()))
        self.fc2_trainer = optim.Adam(list(self.fc2.parameters()) + list(self.rec_fc2.parameters()))
        self.fc3_trainer = optim.Adam(list(self.fc3.parameters()) + list(self.rec_fc3.parameters()))

        self.clf_trainer = optim.Adam(self.classifier.parameters())
    
    @execution_time
    def train(self, epochs = 7):
        print("Train first layer...")
        for epoch in range(epochs):
          for data, _ in self.train_loader:
              data = data.to(self.device)
              out = self.fc1(data)
              rec = self.rec_fc1(out)
              loss = self.predictive_criterion(rec, data)
              self.fc1_trainer.zero_grad()
              loss.backward()
              self.fc1_trainer.step()
          print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("Train second layer...")
        for epoch in range(epochs):
            for data, _ in self.train_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out1 = self.fc1(data)
                out = self.fc2(out1)
                rec = self.rec_fc2(out)
                loss = self.predictive_criterion(rec, out1)
                self.fc2_trainer.zero_grad()
                loss.backward()
                self.fc2_trainer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("Train third layer...")
        for epoch in range(epochs):
            for data, _ in self.train_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out1 = self.fc1(data)
                    out2 = self.fc2(out1)
                out = self.fc3(out2)
                rec = self.rec_fc3(out)
                loss = self.predictive_criterion(rec, out2)
                self.fc3_trainer.zero_grad()
                loss.backward()
                self.fc3_trainer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("Train classifier...")
        for epoch in range(epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    if self.sum_views or self.recurrent:
                        bs = data.size(0)
                        embedding_ori = self.fc3(self.fc2(self.fc1(data)))
                        data_left = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28)).view(bs, -1)
                        embedding_left = self.fc3(self.fc2(self.fc1(data_left)))
                        data_right = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28), angle=-30).view(bs, -1)
                        embedding_right = self.fc3(self.fc2(self.fc1(data_right)))

                        if self.sum_views:
                            embedding = embedding_ori + embedding_left + embedding_right
                        else:
                            embedding = torch.stack([embedding_ori, embedding_left, embedding_right], dim=1)
                    else:
                        embedding = self.fc3(self.fc2(self.fc1(data)))
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
                embedding_ori = self.fc3(self.fc2(self.fc1(data)))
                data_left = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28)).view(bs, -1)
                embedding_left = self.fc3(self.fc2(self.fc1(data_left)))
                data_right = rotate_batch_fixed_angle(data.view(bs, 1, 28, 28), angle=-30).view(bs, -1)
                embedding_right = self.fc3(self.fc2(self.fc1(data_right)))

                if self.sum_views:
                    embedding = embedding_ori + embedding_left + embedding_right
                else:
                    embedding = torch.stack([embedding_ori, embedding_left, embedding_right], dim=1)
            else:
                embedding = self.fc3(self.fc2(self.fc1(data)))
            outputs = self.classifier(embedding)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}%')


@click.command()
@click.option("--trainer", "-t", "trainer", type=str, default="base")
@click.option("--sum-views", "-s", "sum_views", is_flag=True)
@click.option("--recurrent", "-r", is_flag=True)
def main(trainer, sum_views, recurrent):
    torch.manual_seed(42)
    trainers = {'base': BaseTrainer, 'autoencoder': AutoEncoderTrainer, 'predictive': PredictiveTrainer}
    ctrainer = trainers[trainer](sum_views=sum_views, recurrent=recurrent)
    ctrainer.train()
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