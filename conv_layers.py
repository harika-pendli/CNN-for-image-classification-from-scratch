import numpy as np
as_strided  = np.lib.stride_tricks.as_strided

# ----------- to get strided view of the input -----------
def get_strided_view(x, filter_shape, stride):
    '''
    Returns strided view of the input
    
    Args:
       input:  A tensor of shape  N * W_in * H_in * C_in
       filter_shape: The shape of the tensor containing filters  (W_f * H_f)
       stride: A 2d vector for stride length in W and H directions (S_w, S_h)
    
    Returns:
       output: output is of shape N * W_out * H_out * W_f * H_f * C_in
       where, W_out = (W_in - W_f)/S_w + 1
              H_out = (H_in - H_f)/S_h + 1
    '''
    S_w, S_h = stride
    W_f, H_f = filter_shape
    N, W_in, H_in, C_in = x.shape
    
    W_out = (W_in - W_f)//S_w + 1
    H_out = (H_in - H_f)//S_h + 1
    
    view_shape = (N, W_out, H_out, W_f, H_f, C_in)
    view_strides = (x.strides[0], S_w*x.strides[1], S_h*x.strides[2], *(x.strides[1:]))

    return as_strided(x, view_shape, view_strides)



#-------------------- CNN layers ---------------------------
class Convolution:
    def __init__(self, num_channels, filter_size, activation, T, b):
        '''
        Initialise a convoluton layer with given parameters
        '''
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.activation = activation
        self.T = T
        self.b = b

    def forward(self, x):
        '''
        forward pass of convolution
        
        Args:
            x : input is of shape N * W_in * H_in * C_in
            where, N is the batch size
                   W_in * H_in is the size
                   C_in = no. of input channels
            
        Returns:
            output: output is of shape N * W_out * H_out * C_out
            where, W_out = (W_in - W_f)/S_w + 1
                   H_out = (H_in - H_f)/S_h + 1
                   C_out = N_f (no. of filters)
        '''
        # get strided view
        x_view = get_strided_view(x, self.filter_size, (1, 1))
        # multiply with the filter tensor and add bias
        output = np.einsum('nwhklm, cklm -> nwhc', x_view, self.T) + self.b
        # apply activation if required
        if self.activation == 'relu':
            output = relu(output)
        return output

   
class Pooling:
    def __init__(self, dim, type = 'max'):
        '''
        Initialise a pooling layer with given parameters
        '''
        self.dim = dim     # (W_f, H_f)
        self.type = type   # 'max' or 'avg'
 
    def forward(self, x):
        '''
        forward pass of pooling layer
        
        Args:
            x : input is of shape N * W_in * H_in * C_in
            where, N is the batch size
                   W_in * H_in is the size
                   C_in = no. of input channels
            
        Returns:
            output: output is of shape N * W_out * H_out * C_out
            where, W_out = (W_in - W_f)/W_f + 1
                   H_out = (H_in - H_f)/H_f + 1
                   C_out = C_in
        '''
        # get strided view
        x_view = get_strided_view(x, self.dim, self.dim)
        # max pool or avg pool
        if self.type == 'max':
            return x_view.max(axis = (3, 4))
        elif self.type == 'avg':
            return x_view.mean(axis = (3, 4))

    
class FC_sigmoid:
    def __init__(self, b, T):
        '''
        Initialise a fully connected layer with given parameters
        '''
        self.b = b
        self.T = T

    def forward(self, x):
        N, W_in, H_in, C_in = x.shape
        x = x.reshape(N, W_in*H_in*C_in)
        out = x @ self.T + self.b
        out = sigmoid(out)
        return out


# -------------- activation functions ------------------
def relu(x):
    return np.where(x < 0, 0, x)

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))  
