import os
import torch
import pandas as pd
from torch import nn
from torch.optim import SGD
from typing import Tuple


class CarPricePredictor:
    def __init__(self, data_path: str, learning_rate: float = 0.001, epochs: int = 2500) -> None:
        """
        Initializes the model, loss function, optimizer, and loads data.
        
        Formula For  z score normalization :
        
                z = (X – μ) / σ
        where,
        
        X is the value of the data point
        μ is the mean of the dataset
        σ is the standard deviation of the dataset

        """
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Load and preprocess data
        self.X, self.Y = self.load_and_preprocess_data()
        
        # Normalize data
        self.x_mean, self.x_std = self.X.mean(axis=0), self.X.std(axis=0)
        self.y_mean, self.y_std = self.Y.mean(), self.Y.std()
        self.X = (self.X - self.x_mean) / self.x_std
        self.Y = (self.Y - self.y_mean) / self.y_std
        
        # Define model, loss function, and optimizer
        self.model = nn.Linear(3, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)

    def load_and_preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the CSV file and preprocesses the data.
        Returns:
            X (torch.Tensor): Feature matrix
            Y (torch.Tensor): Target values
        """
        df = pd.read_csv(self.data_path)
        
        # Process price
        price = df['price'].str.replace("$", "").str.replace(",", "").astype(int)
        
        # Compute age of the car
        age = df['model_year'].max() - df['model_year']
        
        # Process accident data
        accident_free = (df["accident"] == "None reported").astype(int)
        
        # Process mileage
        mileage = df['milage'].str.replace(",", "").str.replace(" mi.", "").astype(int)
        
        # Convert to tensors
        X = torch.column_stack([
            torch.tensor(accident_free, dtype=torch.float32),
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(mileage, dtype=torch.float32)
        ])
        Y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
        
        return X, Y
    
    def train(self) -> None:
        """
        Trains the model using gradient descent.
        """
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            predictions = self.model(self.X)
            loss = self.loss_function(predictions, self.Y)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    def save_model(self, save_dir: str = "./model") -> None:
        """
        Saves the trained model and normalization parameters.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(self.x_mean, os.path.join(save_dir, 'x_mean.pt'))
        torch.save(self.y_mean, os.path.join(save_dir, 'y_mean.pt'))
        torch.save(self.x_std, os.path.join(save_dir, 'x_std.pt'))
        torch.save(self.y_std, os.path.join(save_dir, 'y_std.pt'))
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

    def predict(self, x_data: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions on new input data.
        Args:
            x_data (torch.Tensor): Input features (accident-free, age, mileage)
        Returns:
            torch.Tensor: Predicted prices
        """
        x_data = (x_data - self.x_mean) / self.x_std  # Normalize
        predictions = self.model(x_data)
        return predictions * self.y_std + self.y_mean
    

if __name__ == "__main__":
    predictor = CarPricePredictor(data_path='./data/used_cars.csv')
    predictor.train()
    predictor.save_model()
    
    # Example predictions
    test_data = torch.tensor([
        [1, 5, 10000],
        [1, 2, 10000],
        [1, 5, 20000]
    ], dtype=torch.float32)
    
    print(predictor.predict(test_data))

    test_data2 = torch.tensor([
        [0, 5, 10000],
        [0, 2, 10000],
        [0, 5, 20000]
    ], dtype=torch.float32)

    print(predictor.predict(test_data2))