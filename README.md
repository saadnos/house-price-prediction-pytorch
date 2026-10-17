# House Price Prediction 

A simple machine learning project where I built a neural network using **PyTorch** to predict house prices based on their features.

## What I Used

- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Matplotlib

## How It Works

- Load and prepare the housing dataset.
- Normalize the input features and house prices.
- Train a neural network using Adam and MSE loss.
- Plot the training loss to see how the model learns.
- Test the model on unseen data and compare predictions with actual prices.

## Model Architecture

The model is a feedforward neural network with two hidden layers, using ReLU activation.

- Input layer: 12 features
- Hidden layers: 20 and 25 neurons
- Output layer: 1 predicted price

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure `data.csv` is in the project folder, then run:

```bash
python train.py
```

The script trains the model, displays the loss graph, prints some predictions, and saves the model weights to `house_model.pth`.

## What I Learned

- Building a neural network with PyTorch
- Training a model using backpropagation
- Preparing and normalizing data
- Evaluating predictions on unseen data
- Visualizing training loss

## Future Improvements

- Add evaluation metrics such as MAE and RMSE.
- Improve prediction accuracy.
- Use the trained model to make predictions without retraining.

---

Built as a hands-on project to learn the fundamentals of deep learning with PyTorch.
