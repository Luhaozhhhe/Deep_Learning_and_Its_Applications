import torch
import time

class Trainer:

    def __init__(self, model, criterion, optimizer, device):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train(self, train_loader, valid_loader, num_epochs):

        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []
        for epoch in range(num_epochs):
            start_time = time.time()  # Start Time

            self.model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            total = 0
            for i, (images, labels) in enumerate(train_loader):
                batch_size = labels.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_size
                total += batch_size

                _, predictions = torch.max(outputs, 1)
                train_accuracy += (predictions == labels).sum().item()

            train_loss /= total
            train_accuracy /= total

            valid_loss, valid_accuracy = self.evaluate(valid_loader, verbose=False)

            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)

            end_time = time.time()  # End Time
            elapsed_time = end_time - start_time  # Elapsed Time

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, '
                  f'Valid Loss: {valid_loss:.4f}, '
                  f'Valid Accuracy: {valid_accuracy:.4f}, '
                  f'Time: {elapsed_time:.2f}s')

        print('Training Finished')
        return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list

    def evaluate(self, test_loader, verbose=False):

        test_loss = 0.0
        test_accuracy = 0.0
        total = 0
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                batch_size = labels.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * batch_size
                total += batch_size

                _, predictions = torch.max(outputs, 1)
                test_accuracy += (predictions == labels).sum().item()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= total
        test_accuracy /= total

        if verbose:
            print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))


        return test_loss, test_accuracy
