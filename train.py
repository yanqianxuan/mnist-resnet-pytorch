import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os

from resnet_model import ResNetMNIST, BasicBlock
from dataset import get_dataloader
from config import get_args

def train():
    args = get_args()
   
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    
    writer = SummaryWriter(log_dir='runs/mnist_experiment')
    
 
    train_loader, val_loader = get_dataloader(args.batch_size)
    
   
    model = ResNetMNIST(block=BasicBlock, layers=[2, 2, 2, 2]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    print(f"🚀 启动成功！正在使用设备: {device}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
   
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
  
        log_msg = f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%"
        print(f"\n{log_msg}")
        with open("train.log", "a") as f:
            f.write(log_msg + "\n")
        
  
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
            
    
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"✨ 发现更好的模型，已保存！")

    writer.close()
    print("✅ 所有训练任务已完成！")

if __name__ == "__main__":
    train()