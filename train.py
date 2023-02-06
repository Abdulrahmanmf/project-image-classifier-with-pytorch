import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms, models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sb
import json
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--arch', type = str, default='vgg16')
    parser.add_argument('--epochs', action='store', type=int, default=3)
    parser.add_argument('--dropout', action='store', type=int, default=0.4)
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', type=int, default=1024)
    parser.add_argument('--data_dir', action='store', default='flowers')
    parser.add_argument('--save_dir', action='store', default='./checkpoint.pth')
    parser.add_argument('--gpu', action="store", default=True)
    
    args = parser.parse_args()
    return args 


def test(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))])

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    return testloader


def my_training_model(criterion, optimizer, validloader, trainloader, my_model, device, epochs ):
    step = 0
    run_loss = 0
    print_for_every = 5

    for any_e in range(epochs):
        for train_in, train_lab in trainloader:
            step += 1
            train_lab,train_in=train_lab.to(device),train_in.to(device)
            optimizer.zero_grad() 

            log_out = my_model.forward(train_in)
            loss = criterion(log_out, train_lab)
            loss.backward() 

            optimizer.step() 
            run_loss += loss.item() 

            if (step % print_for_every == 0): 
                loss_in_valid = 0
                accuracy_in_valid = 0
                my_model.eval() 
                with torch.no_grad(): 
                    for val_in, val_lab in validloader:
                        val_lab,val_in=val_lab.to(device),val_in.to(device)
                        log_out = my_model.forward(val_in)


                        loss_in_batch  = criterion(log_out, val_lab)
                        loss_in_valid += loss_in_batch

                        prob = torch.exp(log_out)
                        top_predicted , top_class = prob.topk(1, dim=1)
                        equals = top_class == val_lab.view(*top_class.shape) 
                        accuracy_in_valid += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch==> {any_e+1}/{epochs}.. "f"Loss_In_Training==> {run_loss/print_for_every:.3f}-- "f"loss_In_Valid==>                                   {loss_in_valid/len(validloader):.3f}-- "f"Accuracy_For_Valid==> {accuracy_in_valid*100/len(validloader):.3f}")
                run_loss = 0
                my_model.train()
    print("Done1")
                                
    return my_model


def main():
    args = parse_args()
    

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))])
    
    train_data  = datasets.ImageFolder(train_dir, transform=train_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32 , shuffle=True)
    
    validloader = test(valid_dir)
    testloader = test(test_dir)
    
    

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_model = getattr(models, args.arch)(pretrained=True)

    for param in my_model.parameters():
        param.requires_grad = False
    in_feature = my_model.classifier[0].in_features
    

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_feature, args.hidden_units)),
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(args.dropout)), 
                                        ('fc2', nn.Linear(args.hidden_units, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))
    my_model.classifier = classifier
    
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.classifier.parameters(), args.learning_rate)
    my_model.to(device);


    my_model = my_training_model(criterion, optimizer, validloader, trainloader, my_model , device, args.epochs)
                      
    lenn = len(testloader)
    loss_in_test = 0
    accuracy_in_test = 0
    my_model.eval()

    with torch.no_grad():
        for test_in, test_lab in testloader:
            test_lab,test_in=test_lab.to(device),test_in.to(device)

            log_out = my_model.forward(test_in)
            loss_in_test += criterion(log_out, test_lab).item()  

       
            prob = torch.exp(log_out)
            top_predicted , top_class = prob.topk(1, dim=1)
            equals = top_class == test_lab.view(*top_class.shape) 
            accuracy_in_test += torch.mean(equals.type(torch.FloatTensor)).item()

    result_test = loss_in_test/lenn
    result_accuracy = 100*accuracy_in_test/lenn
    print(f"The result of the test_loss is ==> {result_test:.2f}--- "
        f"The Best Validation Accuracy==> {result_accuracy:.2f} ")

    my_model.class_to_idx = train_data.class_to_idx 

    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs': args.epochs ,
              'dropouts':args.dropout,
              'learning_rate': learning_rate,
              'classifier': my_model.classifier,
              'class_to_idx': my_model.class_to_idx,
              'optimizer_dict': optimizer.state_dict(),
              'state_dict': my_model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    print("Done!")


if __name__ == "__main__":
    main()




