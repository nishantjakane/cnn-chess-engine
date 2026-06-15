
def train_one_epoch(model,loader,optimizer,criterion,device):
    
    model.train()

    running_loss = 0.0
    
    for boards,targets in loader:
        boards = boards.to(device)
        targets = targets.to(device)

        predictions = model(boards)

        loss = criterion(predictions,targets.unsqueeze(1))
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        running_loss+=loss.item()

    avg_loss = running_loss/len(loader)
    
    return avg_loss

def evaluate(model,loader,criterion,device):
    model.eval()

    running_loss =0

    with torch.no_grad():

        for boards,targets in loader:
            boards = boards.to(device)
            targets = targets.to(device)

            predictions = model(boards)

            loss = criterion(predictions,targets.unsqueeze(1))

            running_loss +=loss.item()
    
    avg_loss = running_loss/len(loader)

    return avg_loss

num_epochs = 10

for epochs in range(num_epochs):

    train_loss = train_one_epoch(model,train_loader,optimizer,criterion,device)

    val_loss = evaluate(model,test_loader,criterion,device)

    print(f"Epoch: {epochs+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")