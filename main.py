import imageio
import numpy as np
from model_utils import *

def perform_classification(task, images):
    
    W, H, N = images.shape
    
    # model expects inputs of shape N * W * H * C_in
    images = np.swapaxes(images, 0, -1)
    images = np.expand_dims(images, -1)
    
    # create the model with required shape
    model = create_model((W, H))
    
    if task == 4:
        # laplacian filter for edge detection
        T = np.array([[[0, 1, 0], 
                       [1,-4, 1], 
                       [0, 1, 0]]])
        b = [0]
        add_conv_layer(model, num_channels = 1,  
                              filter_size = (3, 3), 
                              activation = 'relu', 
                              T = np.expand_dims(T, -1), 
                              b = b)
        add_pooling_layer(model, dim = (125, 125), type = 'avg')
        add_FC_sigmoid_layer(model, b = [3], T = np.array([[-1]]))
    
    
    if task == 2:
        T = np.array([[[0, 0, 0], 
                       [0, 1, 0], 
                       [0, 0, 0]]])/255
        b = [-0.65]
        add_conv_layer(model, num_channels = 1,  
                              filter_size = (3, 3), 
                              activation = 'relu', 
                              T = np.expand_dims(T, -1), 
                              b = b)
        add_pooling_layer(model, dim = (125, 125), type = 'avg')
        add_FC_sigmoid_layer(model, b = -10, T = np.array([1000]))
    
    
    if task == 3:
        
        T = np.array([[[1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1]]])/(255*9)
        b = [-0.9]
        add_conv_layer(model, num_channels = 1,  
                              filter_size = (5, 5), 
                              activation = 'relu', 
                              T = np.expand_dims(T, -1),
                              b = b)
        add_pooling_layer(model, dim = (123, 123), type = 'max')
        add_FC_sigmoid_layer(model, b = [5], T = np.array([[-100]]))
        
    result = model.predict(images)
    classes = np.where(result.ravel() < 0.5, 0, 1)
    return classes


if __name__ == '__main__':
    path = './Images/PS2_Images_Task_1_Class_0/Image'
    images = []
    for i in range(1, 20):
        img = imageio.imread(path + str(i))
        images.append(img)
    images = np.array(images)
    images = np.swapaxes(images, 0, -1)
    classes = perform_classification(4, images)
    print(classes)

