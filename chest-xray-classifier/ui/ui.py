import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn


model = models.resnet50(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model.load_state_dict(torch.load("./chest-xray-classifier/models/model1.pth"))
model.to("cuda" if torch.cuda.is_available() else "cpu")

def infer(model, image, transform):
    model.eval()
    image = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        classes = ["COVID19","NORMAL","PNEUMONIA","TURBERCULOSIS"]
        class_name = classes[predicted.item()]
    print(f"Predicted class: {class_name}")
    return class_name

def prediction(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return infer(model, image, transform)

with gr.Blocks() as demo:
    gr.Markdown("Chest X-RAY Classifier")
    image = gr.Image()
    button = gr.Button("Generate", variant="primary")
    textbox = gr.Text("")

    button.click(
        prediction,
        inputs= image,
        outputs= textbox
    )

if __name__ == "__main__":
    demo.launch(share=True)

