import torch
from torch import nn
from typing import Tuple

class UsedCarPricePredictor:
    def __init__(self, model_path: str = "./model/model.pt") -> None:
        """
        Initializes the predictor by loading the trained model and normalization data.
        """
        self.model = self._load_model(model_path)
        self.x_mean, self.y_mean, self.x_std, self.y_std = self._load_normalization_data()
    
    def _load_model(self, model_path: str) -> nn.Linear:
        """
        Loads the trained model from a file.
        """
        model = nn.Linear(3, 1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    def _load_normalization_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads normalization parameters from saved files.
        """
        x_mean = torch.load("./model/x_mean.pt", weights_only=True)
        y_mean = torch.load("./model/y_mean.pt", weights_only=True)
        x_std = torch.load("./model/x_std.pt", weights_only=True)
        y_std = torch.load("./model/y_std.pt", weights_only=True)
        return x_mean, y_mean, x_std, y_std
    
    def predict(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Predicts the price of used cars based on input values.
        """
        normalized_data = (input_values - self.x_mean) / self.x_std
        prediction = self.model(normalized_data)
        return prediction * self.y_std + self.y_mean

if __name__ == "__main__":
    predictor = UsedCarPricePredictor()
        
    
    print("Predict The Used Car Price by Entering Info:")
    
    try:
        mileage = int(input('Mileage Of Car (in number):'))
        accident = 1 if input("Accident History (type y/n anything outside y or n consider as n):").lower() == 'n' else 0
        age = int(input("Age of the car (in number):"))
        
        
        
        test_values = torch.tensor([accident, age, mileage],dtype=torch.float32)
        predictions = predictor.predict(test_values)
    
        for i, price in enumerate(predictions, 1):
            print(f"\nPredicted Price = ${price.item():,.2f}")

        
    except Exception as e:
        print("Exception Occur:", e)
        print("Enter the detail properly as mentioned")
    
    
    #  Testing
    # test_values = torch.tensor([
    #     [1, 5, 10000],
    #     [1, 2, 10000],
    #     [1, 5, 20000]
    # ], dtype=torch.float32)
    
    # predictions = predictor.predict(test_values)
    
    # for i, price in enumerate(predictions, 1):
    #     print(f"Car {i}: Predicted Price = ${price.item():,.2f}")
