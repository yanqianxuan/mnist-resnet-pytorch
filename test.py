import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from resnet_model import ResNetMNIST, BasicBlock

def predict(image_path):
   
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   
  
    model = ResNetMNIST(block=BasicBlock, layers=[2, 2, 2, 2])
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
  
   
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
       
        transforms.RandomInvert(p=1.0), 
       
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    

    with torch.no_grad():
        output = model(image_tensor)
        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        confidence = torch.max(prob).item()
        
    print(f"图片路径: {image_path}")
    print(f"预测结果: {pred}")
    print(f"置信度: {confidence:.2%}")

if __name__ == "__main__":
    test_img = "test_image.png" 
    if os.path.exists(test_img):
        predict(test_img)
    else:
        print(f"❌ 未找到测试图片: {test_img}，请先在文件夹放一张图片。")