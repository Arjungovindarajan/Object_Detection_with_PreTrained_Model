import torch
from model import get_model
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(device, model_path='insulator_detector.pth'):
    # Load the model
    model = get_model(num_classes=2)  # Assuming 2 classes: background, insulator
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def predict_and_display(model, img_path, device):
    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)  # Add batch dimension

    # Run the model
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract bounding boxes and scores
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Filter predictions based on a score threshold
    score_threshold = 0.5
    boxes = boxes[scores >= score_threshold]

    # Display the image with bounding boxes
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Add the bounding boxes to the image
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Example usage:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = load_model(device)
predict_and_display(model, 'C:/Users/CD-9/Downloads/Experiment_dataset-20241021T075135Z-001/Experiment_dataset/Images/27.jpg', device)
