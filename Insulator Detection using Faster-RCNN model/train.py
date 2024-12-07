import torch.optim as optim
import torch
def train_model(model, data_loader, device, num_epochs=10):
    model.train()
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    
    torch.save(model.state_dict(), 'insulator_detector.pth')
    print("Model saved to insulator_detector.pth")
# Use GPU if available
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
