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



def get_OutArgument():
   
    
    parser = argparse.ArgumentParser(description = 'outimage')
    parser.add_argument('--checkpoint', type = str ,  default = './checkpoint.pth', action="store")
    parser.add_argument('--category_names', default='cat_to_name.json' ,  type = str)
    parser.add_argument('--images', type = str , action="store" )
    parser.add_argument('--top_k', type = int, action='store', default=5)
    parser.add_argument('--gpu', type = str, action='store', default='gpu')
    args = parser.parse_args()
    return args


def predict(images , my_model , cat_to_name, topk , device):
    
    my_model.to(device) 
    my_model.eval()
    pic = process_image(images) 
    pic = torch.from_numpy(pic).type(torch.FloatTensor) 
    pic = pic.unsqueeze_(0)  
    
    with torch.no_grad():
        log_out = my_model.forward(pic)
        prob = torch.exp(log_out)
        top_predicted, top_class = prob.topk(topk) 
        np_top_class = np.array(top_class)
        
        idx_to_class = {val: key for key, val in my_model.class_to_idx.items()}
        top_predicted = np.array(top_predicted)[0] 
        top_class = np.array(top_class)[0]
        top_class = [idx_to_class[i] for i in top_class] 
        top_flowers = [cat_to_name[i] for i in top_class] 
    
    
    return top_predicted, top_class, top_flowers
    
def process_image(image):
    standard_deviations=[0.229, 0.224, 0.225]
    means=[0.485, 0.456, 0.406]
    img = Image.open(image)
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(standard_deviations,means)])
    transpic = trans(img)
    return (np.array(transpic))

def main():
    args = get_OutArgument()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'

    if args.gpu and not (torch.cuda.is_available()):
        device = 'cpu'
        print("GPU isn't available. CPU is chosen")
    else:
        device = 'cpu'

        
    checkpoint = torch.load(args.checkpoint)

    my_model = my_models.vgg16(pretrained=True)

    for param in my_model.parameters():
        param.requires_grad = False
    
    my_model.class_to_idx = checkpoint['class_to_idx']
    my_model.classifier = checkpoint['classifier']
    my_model.load_state_dict(checkpoint['state_dict'])
    

    P_im = process_image(args.images)

    top_predicted, top_flowers = predict(P_im, my_model, args.top_k, cat_to_name, device)

    topflowers=top_flowers[0]
    toppredicted=top_predicted[0]
    print("Toppest --{}-- flower classes to predict ==> {}".format(args.top_k, top_flowers))
    print("Toppest class of flower==> {}".format(topflowers))
    print("Toppest --{}-- predict of values==> {}".format(args.top_k, top_predicted))
    print("acc ==> {:.3}".format(100 * toppredicted))
    
    
    

if __name__ == "__main__":
    main()  