# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:00:47 2020

@author: L430
"""
from __future__ import print_function, division
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import os
import torch.nn.functional as F
from torch import nn
from torch import optim
transform=transforms.Compose([ torchvision.transforms.Grayscale(num_output_channels=1), #definir las trasnformaciones de las imagenes
                             torchvision.transforms.Resize((28,28), interpolation=2),#Asigna el tamaño de 28,28
                             transforms.RandomRotation(30), # Rota la imagen 30°
                             transforms.RandomHorizontalFlip(), #mueve la imagen Horizontalmente
                             transforms.ToTensor()]) #la imagen la convierte a un tensor
data_dir = './data/' #Entramos al directorio de las imagenes


####################################################3


def view_classify(img, ps, version="data"):
    ''' Funcion para ver una imagen y las clases predichas
        y la grafica que dice a que clase se parece mas
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(3), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(3))
    if version == "data":
        
        ax2.set_yticklabels(['CINCUENTA',
                            'CIEN',
                            'DOSIENTOS',
                            'QUNIENTOS',], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()



###################################################

imageFolder = {} #crear una carpeta para almecenar las imagenes cargadas
for r in ['train', 'test']:#for para entrar a las carpetas de entrenamiento y prueba
    x = os.path.join(data_dir, r)#Declara una variable para recorrer las imagenes
    imageFolder[r] = torchvision.datasets.ImageFolder(root=x, transform=transform) #Entramos a la carpeta de las y las carga al dataLoader

dataLoaders = {} #Carga las imagenes
for i in ['train', 'test']: #for para recorrer las imagenes de las carpeta de entrenamiento y prueba
    dataLoaders[i] = torch.utils.data.DataLoader(imageFolder[i], #dataLoader = generador
               batch_size=6, shuffle=True, num_workers=1) #generador con un llote de 5 y shuffle barajea las imagenes
def imshow(inp, title = None): # funcion para mostrar las imagenes
    inp = inp.numpy().transpose((1,2,0))#Transpone la posicion de las imagenes
    plt.imshow(inp) # muestra las imagenes
    if title is not None:
        plt.title(title) #muestra las etiquetas de las imagnees
input, classes = next(iter(dataLoaders['train'])) #muestra las clases del train

out = torchvision.utils.make_grid(input) #manda una salida de las imagenes del train
class_names = imageFolder['train'].classes #muestra las clases de las imagenes del train
imshow(out, title = [class_names[x] for x in classes]) #imprime las imagens con sus etiquetas

#convierte a modelos lineales 
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(), #lo hace modelo lienal si es menor a 0 lo hace cero y va a 1
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

#DEfinir las perdidas
criterion = nn.NLLLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.003) 
#defiminmos las epocas que se desea realizar para mostrar las perdidas
epochs = 200 #dosientas epocas
for e in range(epochs): #for para crear las epocas
    running_loss = 0
    for images, labels in dataLoaders[i]:
        #  Acoplar imágenes de dara en un vector
        images = images.view(images.shape[0], -1)
    
        # TODO: pase de entrenamiento
        optimizer.zero_grad() #optimiza
        
        output = model(images) #muestra la salida de las imagenes
        loss = criterion(output, labels) #muestra las perdidas 
        loss.backward()
        optimizer.step()#optimiza
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(dataLoaders)}") #muestra cada una de las perdidas dependiendo las epocas

import helper #importamos la libreria helper

images, labels = next(iter(dataLoaders[i]))#carga las imagenes

img = images[0].view(1, 784) #visualizar las imagenes
# apaga los gradientes para acelarar el processo
with torch.no_grad():
    logps = model(img)

# La salida de la red son log-probabilidades, debe tomar exponencial para las probabilidades
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps) #clasifica las imagenes


