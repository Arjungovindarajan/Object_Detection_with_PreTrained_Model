import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load the pretrained Faster R-CNN model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the head with a new one (number of classes + background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model