import torch
import model
import torch.optim as optim

def train_model(model_, criterion, optimizer_ft, num_epochs=25):
    return model_

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    localizer = model.Localizer()
    model_ = localizer.getModel()
    model_ = model_.to(device)
    criterion = torch.nn.MSELoss() 
    optimizer_ft = optim.Adam(model_.parameters(),lr=0.001)
    model_ft = train_model(model_, criterion, optimizer_ft, num_epochs=25)
    #print(model_ft)

if __name__ == "__main__":
    main()