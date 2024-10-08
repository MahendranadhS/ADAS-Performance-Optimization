import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval() 

# Define the image transformation function
transform = T.Compose([T.ToTensor()])

# Load and transform an image from the KITTI dataset
def load_image(img_path):
    image = Image.open(img_path)
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)  # Add batch dimension

# Perform object detection
def detect_objects(model, img_tensor):
    with torch.no_grad():
        predictions = model(img_tensor)
    return predictions

# Display the image and bounding boxes
def display_image_with_boxes(ax, img_path, predictions, threshold=0.5):
    image = Image.open(img_path)
    ax.clear()  # Clear the previous image
    ax.imshow(image)
    
    # Filter predictions with scores higher than the threshold
    for elem in range(len(predictions[0]['boxes'])):
        if predictions[0]['scores'][elem] > threshold:
            box = predictions[0]['boxes'][elem].numpy()
            # Draw bounding boxes
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                       edgecolor='red', fill=False, linewidth=2))
    ax.axis('off')  # Hide axes

# Process images with navigation
def process_images_with_navigation(image_directory):
    # List all PNG files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    total_images = len(image_files)

    fig, ax = plt.subplots()
    fig.suptitle("Object Detection on KITTI Images")
    
    current_image_index = [0]  # Use a list to make it mutable in the callback

    # Function to update the image
    def update_image():
        img_path = os.path.join(image_directory, image_files[current_image_index[0]])
        img_tensor = load_image(img_path)
        predictions = detect_objects(model, img_tensor)
        display_image_with_boxes(ax, img_path, predictions)
        plt.draw()  # Refresh the plot

    # Button click handler
    def next_image(event):
        if current_image_index[0] < total_images - 1:
            current_image_index[0] += 1
            update_image()

    # Create a button for navigation
    ax_next = plt.axes([0.81, 0.01, 0.1, 0.05])
    btn_next = plt.Button(ax_next, 'Next Image')
    btn_next.on_clicked(next_image)

    # Display the first image
    update_image()
    
    plt.show()

image_directory = 'D:/IT/Coding Workspace/Projects/ADAS_Peformance_Optimization/images/image_00/Data'
process_images_with_navigation(image_directory)
