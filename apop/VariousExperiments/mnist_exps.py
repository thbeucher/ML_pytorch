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
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(32, 10))
    
    def forward(self, x):
        return self.fc(x)


class BaseTrainer():
    def __init__(self):
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

    @execution_time
    def train(self, epochs=10):
        for epoch in range(epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
    
    def create_model(self):
        self.fc1 = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

        self.rec_fc1 = nn.Sequential(nn.Linear(128, 28*28), nn.Sigmoid())
        self.rec_fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.rec_fc3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())

        self.classifier = Classifier()
    
    def get_criterion_n_optimizer(self):
        self.predictive_criterion = nn.MSELoss()
        self.clf_criterion = nn.CrossEntropyLoss()

        self.fc1_trainer = optim.Adam(list(self.fc1.parameters()) + list(self.rec_fc1.parameters()))
        self.fc2_trainer = optim.Adam(list(self.fc2.parameters()) + list(self.rec_fc2.parameters()))
        self.fc3_trainer = optim.Adam(list(self.fc3.parameters()) + list(self.rec_fc3.parameters()))

        self.clf_trainer = optim.Adam(self.classifier.parameters())
    
    def train(self):
        print("Train first layer...")
        epochs = 7
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
                    out1 = self.fc1(data)
                    out2 = self.fc2(out1)
                    out3 = self.fc3(out2)
                out = self.classifier(out3)
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
            out1 = self.fc1(data)
            out2 = self.fc2(out1)
            embedding = self.fc3(out2)
            outputs = self.classifier(embedding)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Classifier Accuracy: {accuracy:.2f}%')


@click.command()
@click.option("--trainer", "-t", "trainer", type=str, default="base")
def main(trainer):
    trainers = {'base': BaseTrainer, 'autoencoder': AutoEncoderTrainer, 'predictive': PredictiveTrainer}
    ctrainer = trainers[trainer]()
    ctrainer.train()
    ctrainer.evaluate()


if __name__ == "__main__":
    main()