import cv2
import numpy as np
import torch

# Function to draw bounding boxes on an image
def draw_bboxes(image, boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    return image

# Evaluation function to display images with predictions
def evaluate_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for imgs, targets in data_loader:  # Use a loop, don't index data_loader
            imgs = [img.to(device) for img in imgs]

            # Forward pass
            predictions = model(imgs)

            for i, pred in enumerate(predictions):
                # Extract boxes and filter by score threshold (optional)
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                score_threshold = 0.5
                boxes = boxes[scores >= score_threshold]

                # Convert image tensor to a format for OpenCV
                img = imgs[i].cpu().permute(1, 2, 0).numpy()  # Convert CHW to HWC
                img = (img * 255).astype(np.uint8)  # Convert to uint8 for OpenCV

                # Draw bounding boxes on the image
                img_with_boxes = draw_bboxes(img, boxes)

                # Display the result using OpenCV
                cv2.imshow(f"Image {i} - Detected Insulators", img_with_boxes)
                cv2.waitKey(0)  # Press any key to close the image

                print(f"Image {i}: Detected {len(boxes)} insulators")

        # Close all OpenCV windows after evaluation
        cv2.destroyAllWindows()