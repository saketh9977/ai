from datetime import datetime, timedelta

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def print_c(msg):
    time_c = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{time_c} IST - {msg}")

class CData(Dataset):
    def __init__(self, X_train, Y_train):
        self.X=torch.from_numpy(X_train)
        self.Y=torch.from_numpy(Y_train)
        self.len=self.X.shape[0]
    def __getitem__(self,index):      
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.len
    
class CNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNet,self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU() # introduce non-linearity, help mitigate vanishing gradient
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def main():
    print_c("nn: starting")

    # prepare dataset
    X, Y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3
    )
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    data = CData(X_train, Y_train)
    loader = DataLoader(dataset=data, batch_size=64)

    # define NN
    input_dim = 2 # features per data point
    hidden_dim = 25 # hidden layers
    output_dim=3    # total classes in classification task
    clf = CNet(input_dim, hidden_dim, output_dim)
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)

    # training
    num_epochs = 1000
    for epoch in range(num_epochs):
        for batch in loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(torch.float32)
            y_batch = y_batch.to(torch.long)

            y_pred = clf(x_batch)
            loss = loss_f(y_pred, y_batch)
            
            optimizer.zero_grad() # Clear gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update model parameters

            print_c(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")
    
    # testing
    clf.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient computation for inference to save memory
        x_test_tensor = torch.tensor(X_test).to(torch.float32)
        predictions = clf(x_test_tensor)
        predicted_labels = torch.argmax(predictions, dim=1)  # Choose the class with the highest probability
    
    # compute accuracy
    correct_predictions = 0
    for i in range(len(Y_test)):
        if predicted_labels[i] == Y_test[i]:
            correct_predictions += 1
    accuracy = (correct_predictions / (len(Y_test)))*100
    print_c(f"accuracy = {round(accuracy, 2)}%")

    print_c("nn: ending")

if __name__ == '__main__':
    main()