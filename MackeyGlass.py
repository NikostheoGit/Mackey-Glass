import pickle5 as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


with open(r"\Users\Nikos\Downloads\ml_assignment\data\MackeyGlass\train.pickle", "rb") as file:
    train_data = pickle.load(file)['data']
with open(r"\Users\Nikos\Downloads\ml_assignment\data\MackeyGlass\val.pickle", "rb") as file:
    val_data = pickle.load(file)['data']
with open(r"\Users\Nikos\Downloads\ml_assignment\data\MackeyGlass\test.pickle", "rb") as file:
    test_data = pickle.load(file)['data']

print(f'Train data shape: {train_data.shape}')
print(f'Validation data shape: {val_data.shape}')
print(f'Test data shape: {test_data.shape}')

def create_samples(data, look_back):
    X, y = [], []
    for trajectory in data: #976 eikositetrades gia ka8e trajectory Sunolo 97600 eikositetrades
        for i in range(len(trajectory) - look_back):
            X.append(trajectory[i:i + look_back])
            y.append(trajectory[i + look_back])
    return np.array(X), np.array(y)

look_back = 24
X_train, y_train = create_samples(train_data, look_back) 
X_val, y_val = create_samples(val_data, look_back)
#  X_test, y_test are not created

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

# Scaling the data
scaler = MinMaxScaler() # ---->  ! RESHAPE X_train.reshape(-1, X_train.shape[-1]) [97600*24,1] stoixeia
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
y_val = scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
test_data_scaled = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

# Convert to PyTorch tensors
X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=32)

class CNN(nn.Module):
    def __init__(self, input_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_length * 32, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = x.squeeze(-1)  # Remove the last dimension, resulting in shape [batch_size, length]
        x = x.unsqueeze(1)  # Add channel dimension, resulting in shape [batch_size, 1, length]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_model(model, train_loader, val_loader, num_epochs=50, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    mseF=[]
    mse_criterion = nn.MSELoss()  # Define the MSE loss function
    total_loss = 0.0
    count = 0

    for epoch in range(num_epochs):
        model.train() #Activation of Training Mode: (Dropout Layers etc)
        running_loss = 0.0
        for inputs, targets in train_loader: #TRAIN  (inputs[0].shape =torch.Size([32, 24, 1]) [Batch size,L,Dimensionality])
            optimizer.zero_grad() # Clear gradients of all optimized tensors
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Ypologizoume to Loss sto sugkekrimeno Batch
            loss.backward() # Ypologizei tis paragwgous wrt Weights+Bias kai 
                            # autes oi paragwgoi mas deixnoyn pros poia kateu8ynsh prepei
                            #na kinh8oume gia na exoume elaxistopoihsh ths loss function.
                            
            optimizer.step() #Update Weights after each Batch
            running_loss += loss.item() * inputs.size(0)
            
            loss2 = mse_criterion(outputs, targets)
            total_loss += loss2.item() * len(targets)
            count += len(targets)
            
        train_loss = running_loss / len(train_loader.dataset) #/97600
       
        train_losses.append(train_loss)
        
        mse = total_loss / count
        mseF.append(mse)
       
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader: #Validation Set
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict() #Save best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
    
    # Load the best model
    model.load_state_dict(best_model) #Load best model
    return train_losses, val_losses, mseF

# Initialize and train the model
model = CNN(look_back)
train_losses, val_losses, mseF = train_model(model, train_loader, val_loader)

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

def calculate_abs_error(predictions, targets):
    return np.linalg.norm(predictions - targets)     #np.abs((predictions - targets))


def iterative_predictions(model, test_data, look_back, scaler):
    model.eval()
    predictionsItter = []
    predictionsSingle = []
    cc = 0
    initial_inputItter = []
    initial_inputSingle = []# initial input of each trajectory
    
    
    FinalabsListItter=[]
    FinalabsListSingle=[]
    
    FinalmseItter=[]
    FinalmseSingle=[]
    for trajectory in test_data:
        cc += 1
        # Initialize with the first look_back timesteps (Itter)
        X_initialItter = trajectory[:look_back].reshape(1, look_back, 1) #[Batch,L,1]=[1,L,1]
        
        current_trajectory_predsItter = []
        
        initial_inputItter.append(X_initialItter)
        
         # Initialize with the first look_back timesteps (Single)
        X_initialSingle = trajectory[:look_back].reshape(1, look_back, 1)
        current_trajectory_predsSingle = []
        
        initial_inputSingle.append(X_initialSingle)
        
        
        abs_listItter=[0] * look_back
        abs_listSingle=[0] * look_back
        
        true_values = trajectory[look_back:]  # True values for both Auto and Single

        
        for i in range(len(trajectory) - look_back):  # or len(trajectory) - look_back if you want to predict until end
            with torch.no_grad():
                #Itterative
                inputsItter = torch.Tensor(X_initialItter)
                outputItter = model(inputsItter).item()  # Predict next timestep
                current_trajectory_predsItter.append(outputItter)
                
                            
                # Single Step
                inputsSingle = torch.Tensor(X_initialSingle)
                outputSingle = model(inputsSingle).item()  # Predict next timestep
                current_trajectory_predsSingle.append(outputSingle)                
                
                
                abs_stepItter = calculate_abs_error(np.array([outputItter]), np.array([true_values[i]]))
                abs_listItter.append(abs_stepItter.flatten())  # Store MSE for plotting
                
                abs_stepSingle = calculate_abs_error(np.array([outputSingle]), np.array([true_values[i]]))
                abs_listSingle.append(abs_stepSingle.flatten())  # Store MSE for plotting
                

                    
                
            # Update X_initial by appending the predicted value and removing the first element
            
            X_initialItter = np.roll(X_initialItter, -1)  # Shift array left
            X_initialItter[0, -1, 0] = outputItter  # Update the last element with predicted value
            
            # Update X_initial by appending the predicted value and removing the first elemen
            
            X_initialSingle = np.roll(X_initialSingle, -1)  # Shift array left
            X_initialSingle[0, -1, 0] = true_values[i]  # Update the last element with Real Value

        
        
        
        FinalabsListItter.append(abs_listItter)
        FinalabsListSingle.append(abs_listSingle)
        
        predictionsItter.append(current_trajectory_predsItter)
        predictionsSingle.append(current_trajectory_predsSingle)
               

        mseItter=mean_squared_error(true_values,current_trajectory_predsItter)
        mseSingle=mean_squared_error(true_values,current_trajectory_predsSingle, )
        
        FinalmseItter.append(mseItter)
        FinalmseSingle.append(mseSingle)
        
        print('Trajectory Number', cc,'mseItter= ',mseItter,'mseSingle= ',mseSingle)
        


    # Inverse transform the predictions to the original scale
    predictions_unscaledItter = []
    for trajectory_preds in predictionsItter:
        trajectory_preds_unscaledItter = scaler.inverse_transform(np.array(trajectory_preds).reshape(-1, 1)).flatten()
        predictions_unscaledItter.append(trajectory_preds_unscaledItter)

    # Inverse transform the predictions to the original scale
    predictions_unscaledSingle = []
    for trajectory_preds in predictionsSingle:
        trajectory_preds_unscaledSingle = scaler.inverse_transform(np.array(trajectory_preds).reshape(-1, 1)).flatten()
        predictions_unscaledSingle.append(trajectory_preds_unscaledSingle)
        
    return predictions_unscaledItter, initial_inputItter, FinalabsListItter,FinalmseItter,predictions_unscaledSingle,initial_inputSingle,FinalabsListSingle,FinalmseSingle

# Perform iterative predictions on the scaled test set
test_predictionsItter, initial_inputItter ,abs_listItter,FinalmseItter,test_predictionsSingle,initial_inputSingle,abs_listSingle,FinalmseSingle= iterative_predictions(model, test_data_scaled, look_back, scaler)


Traj=88



def plots(Traj):
    #Itterative trajectory predictions
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(test_data[Traj], label='True Data', linestyle='--')
    ax1.plot(scaler.inverse_transform(np.array(initial_inputItter[Traj][0]).reshape(-1, 1)).flatten(), label='Initial Input', marker='.', linestyle='-')
    ax1.plot(np.concatenate(([np.nan]*look_back, test_predictionsItter[Traj])), label='Predictions', marker='.')
    ax1.set_title('Iterative Predictions on Test Data')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Timesteps')
    ax1.legend()
    
    #Error Itterative trajectory predictions
    ax2.plot(abs_listItter[Traj], marker='o', linestyle='None', markersize=2, color='blue')
    ax2.set_ylabel('y_pred - y_real ')
    ax2.set_title('Error for Itterative (autoregressive) Prediction per Timestep')
    ax2.set_xlabel('Timesteps') 
    ax2.grid(True)
    ax2.legend()
    plt.show()
    
    
    
    
    # Single Step trajectory predictions
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(test_data[Traj], label='True Data', linestyle='--')
    ax1.plot(scaler.inverse_transform(np.array(initial_inputSingle[Traj][0]).reshape(-1, 1)).flatten(), label='Initial Input', marker='.', linestyle='-')
    ax1.plot(np.concatenate(([np.nan]*look_back, test_predictionsSingle[Traj])), label='Predictions', marker='.')
    ax1.set_title('Single Test Predictions on Test Data')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Value')
    ax1.legend()
    
    # Plot error Single Step
    ax2.plot(abs_listSingle[Traj],  marker='o', linestyle='None', markersize=2, color='red')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('y_pred - y_real')
    ax2.set_title('Error for Single Step Prediction per Timestep')
    ax2.grid(True)
    ax2.legend()
    plt.show()
    
    
    
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #Error Single vs Error Itterative per Timestep
    ax1.plot(abs_listItter[Traj], color='b', label='Itterative')
    ax1.plot(abs_listSingle[Traj], color='r', label='Single Step')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel(' y_pred - y_real ')
    ax1.set_title('Error Single Step VS Itterative per Timestep')
    ax1.grid(True)
    ax1.legend()
    
    
    ax2.plot(FinalmseItter, marker='.', color='b', label='MSE Error Itterative')
    ax2.plot(FinalmseSingle, marker='.', color='r', label='MSE Error Single Step')
    ax2.set_xlabel('Trajectory')
    ax2.set_ylabel('MSE Error')
    ax2.set_title('MSE Error Single Step VS Itterative per Trajectory')
    ax2.grid(True)
    ax2.legend()
    plt.show()