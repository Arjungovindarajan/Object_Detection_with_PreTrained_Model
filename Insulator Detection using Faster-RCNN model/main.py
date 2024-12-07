from extract_bound_box import InsulatorDataset
# from torchvision.transforms import Compose,ToTensor
from model import get_model
from train import train_model
from eval_mod import evaluate_model
from Load_Model import load_model
from torch.utils.data import DataLoader
import torch

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Device configuration (GPU or CPU)
    # transform = Compose([ToTensor()])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Step 1: Load the dataset
    dataset = InsulatorDataset(root_dir="C:/Users/CD-9/Downloads/Experiment_dataset-20241021T075135Z-001/Experiment_dataset")
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    
    # Step 3: Train the model
    # train_model(model, data_loader, device, num_epochs=10)
    
    # Load the model
    model = load_model(device, 'insulator_detector.pth')

    # Step 4: Evaluate and run inference
    evaluate_model(model,data_loader,device)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()