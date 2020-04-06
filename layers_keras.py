from __future__ import print_function
from custom_model.math_utils import *
import random 
import numpy as np

from keras import activations, initializers, constraints, regularizers
import keras.backend as K
from keras.layers import Layer, Dense, LeakyReLU, Dropout, Reshape, Flatten
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
############################

dim = 199
# construct adjacency matrix and scaled Laplacian for a circle graph with dim nodes
A = adjacency_matrix(dim)
scaled_laplacian = normalized_laplacian(A)

############################
class LocallyConnectedGC(Layer):
    """
    Locally-connected graph convolutional layer
        Arguments:
            k: receptive field, k-hop neighbors
            scaled_lap: scaled Laplacian matrix
    """
    
    def __init__(self, k,
                 scaled_lap=scaled_laplacian,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LocallyConnectedGC, self).__init__(**kwargs)
        
        self.k = k
        self.scaled_lap = scaled_lap
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        self.supports = []
        # k-hop adjacency matrix
        S = K.constant(K.to_dense(calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(S)
        
    def compute_output_shape(self, input_shapes):
        return input_shapes

    def build(self, input_shapes):
        nb_nodes = input_shapes[1]
        feature_dim = input_shapes[-1]
        
        self.kernel = self.add_weight(shape=(nb_nodes, nb_nodes),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(feature_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs
        W = self.supports[0]*self.kernel
        X = K.dot(K.permute_dimensions(features, (0,2,1)), W)
        outputs = K.permute_dimensions(X, (0,2,1))
        if self.use_bias:
            outputs += self.bias
        return self.activation(outputs)

######################################
class Linear(Layer):
    """
    Node-wise dense layer wih given output dimension
        Arguments:
            units: dimension of output
    """
    
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Linear, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    def compute_output_shape(self, input_shapes):
        output_shape = (None, input_shapes[1], self.units)
        return output_shape

    def build(self, input_shapes):
        input_dim = input_shapes[-1]
        self.kernel_lin = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel_lin',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        outputs = K.dot(inputs, self.kernel_lin)
        if self.use_bias:
            outputs += self.bias
        return self.activation(outputs)
    
############################ 
class Scheduled(Layer):
    """
    Scheduled sampling layer, input = [ground_truth, previous_prediction]
        Arguments:
            coin: probability to use previous prediction, defaut = 0
        Methods:
            reset_coin: to change the probability
    """  
    
    def __init__(self, coin=0.,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Scheduled, self).__init__(**kwargs)
        self.coin = coin

    def compute_output_shape(self, input_shapes):
        return input_shapes[-1]

    def build(self, input_shapes):
        self.built = True
    
    def reset_coin(self, s):
        self.coin = s

    def call(self, inputs, mask=None):
        rand = random.uniform(0, 1)
        if rand > self.coin:
            outputs = K.permute_dimensions(inputs[0], (0,2,1))
        else:
            outputs = inputs[1]        
        return outputs

    def get_config(self):
        config = {'coin': self.coin}

        base_config = super(Scheduled, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
##########################
class GraphAttention(Layer):
    """
    Graph Attention layer, Eq-(GAN) in the paper
        Arguments:
            k: receptive field
            F_: dimension of hidden feature of each node
            attn_heads: number of attention heads, defaut = 1
            attn_heads_reduction: methods to combine multi-heads attention,
                {'concat', 'average'}
            dropout_rate: dropout rate for regularization, defaut = 0
            scaled_lap: scaled Laplacian matrix
    """
    
    def __init__(self,
                 k,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.0,
                 scaled_lap = scaled_laplacian,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        
        self.k = k
        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.scaled_lap = scaled_lap
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads
        
        self.supports = []
        s = K.constant(K.to_dense(calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(s)

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        F = input_shape[-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs  # Node features (N x F)
        A  = self.supports[0]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.permute_dimensions(attn_for_neighs, (0,2,1))  # (N x N) via broadcasting

            # Add nonlinearty, optional
            #dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e15 * (1.0 - A)
            dense += mask
            #dense = dense*A

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.batch_dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], self.output_dim)
        return output_shape

##################################
class DynamicGC(Layer):
    """
    Dynamic graph convolutional module, Eq-(5)/(DGC) in the paper
        Arguments:
            k: receptive field
            units: output dimension
            normalization: use softmax normalization or not, defaut = None
            attn_heads=1,
            attn_heads_reduction='concat',  # {'concat', 'average'}
            scaled_lap = scaled_laplacian,
    """

    def __init__(self,
                 k,
                 units,
                 normalization=False,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 scaled_lap = scaled_laplacian,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        
        self.k = k
        self.units = units
        self.normalization = normalization
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.scaled_lap = scaled_lap
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.parameters = []
        self.fc = []
        self.biases = []        # Layer biases for attention heads
        
        self.supports = []
        s = K.constant(K.to_dense(calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(s)

        super(DynamicGC, self).__init__(**kwargs)

    def build(self, input_shape):
        F = input_shape[-1]
        N = input_shape[-2]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, N),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            
            self.kernels.append(kernel)
            # Layer parameters
            parameter = self.add_weight(shape=(N, N),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='parameter_{}'.format(head))
            self.parameters.append(parameter)
            
            transform = self.add_weight(shape=(F, self.units),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='transform_{}'.format(head))
            self.fc.append(transform)

            # # Layer bias
            if self.use_bias:
                bias1 = self.add_weight(shape=(N, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias1_{}'.format(head))
                
                bias2 = self.add_weight(shape=(self.units, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias2_{}'.format(head))
                self.biases.append([bias1, bias2])

        self.built = True

    def call(self, inputs):
        X = inputs  # Node features (N x F)
        A = self.supports[0]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]#[1]  # W in the paper (F x N)
            W = self.parameters[head]
            T = self.fc[head]
            bias_N = self.biases[head][0]
            bias_units = self.biases[head][1]
            
            feature = K.dot(K.permute_dimensions(X, (0,2,1)), A*W)
            dense = K.dot(K.permute_dimensions(feature, (0,2,1)), kernel)
            #dense = K.dot(dense, kernel)
            if self.use_bias:
                dense = K.bias_add(dense, bias_N)
                
            if self.normalization:
                mask = dense + -10e15 * (1.0 - A)
                mask = K.softmax(mask)
            else:
                mask = dense*A

            # Linear combination with neighbors' features
            #node_features = K.batch_dot(mask, feature)
            node_features = K.batch_dot(mask, X)  # (N x F)
            node_features = K.dot(node_features, T) #(N x units)

            if self.use_bias:
                node_features = K.bias_add(node_features, bias_units)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=1)  # N x F')

        output = self.activation(output)
        #return output
        
        # for model interpretation
        return K.concatenate([output, mask], axis=-1)

    def compute_output_shape(self, input_shapes):
        if self.attn_heads_reduction == 'concat': 
            output_shape = (input_shapes[0], input_shapes[1], self.units*self.attn_heads + input_shapes[1])
        else:
            output_shape = (input_shapes[0], input_shapes[1], self.units)
        return output_shape

###########################
class DGCRNNCell(Layer):
    """
    RNN cell with GRU structure and spatial attention gates
    Eq-(DGGRU) in the paper
        Arguments:
            k: receptive field - 1
            dgc_mode: spatial attention module, 
                {'dgc','gan','lc'}
    """
    
    def __init__(self, k, 
                 dgc_mode='dgc',
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DGCRNNCell,self).__init__(**kwargs)
        self.k = k
        self.dgc_mode = dgc_mode
        self.state_size = dim
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.supports_masking = True
    
    def build(self, input_shape):
        #input_shape = (bs, nb_nodes)
        shapes = (input_shape[0], input_shape[1], 2)
        inner = (input_shape[0], input_shape[1], 2*self.k+1)
        shapes_1 = (input_shape[0], input_shape[1], 1)
        
        if self.dgc_mode == 'dgc':
            self.dgc_r = DynamicGC(k=self.k,units=1, activation=self.recurrent_activation)
            self.dgc_r.build(shapes)
            wr_1 = self.dgc_r.trainable_weights

            self.dgc_u = DynamicGC(k=self.k,units=1, activation=self.activation)
            self.dgc_u.build(shapes)
            wu_1 = self.dgc_u.trainable_weights

            self.dgc_c = DynamicGC(k=1,units=1, activation=self.recurrent_activation)
            self.dgc_c.build(shapes)
            wc_1 = self.dgc_c.trainable_weights
            
            
            w = wr_1 + wu_1 + wc_1
            
        if self.dgc_mode == 'gan':
            self.dgc_r = GraphAttention(k=self.k, F_=2*self.k+1)#, depthwise=False
            self.dgc_r.build(shapes)
            wr_1 = self.dgc_r.trainable_weights

            self.lin_r = Linear(1, activation=self.recurrent_activation)
            self.lin_r.build(inner)
            wr_2 = self.lin_r.trainable_weights

            self.dgc_u = GraphAttention(k=self.k, F_=2*self.k+1)#, depthwise=False
            self.dgc_u.build(shapes)
            wu_1 = self.dgc_u.trainable_weights

            self.lin_u = Linear(1, activation=self.recurrent_activation)
            self.lin_u.build(inner)
            wu_2 = self.lin_u.trainable_weights

            self.dgc_c = GraphAttention(1, F_=2*self.k+1)#, depthwise=False
            self.dgc_c.build(shapes)
            wc_1 = self.dgc_c.trainable_weights

            self.lin_c = Linear(1, activation=self.activation)
            self.lin_c.build(inner)
            wc_2 = self.lin_c.trainable_weights
            
            w = wr_1+wr_2+wu_1+wu_2+wc_1+wc_2
            
        if self.dgc_mode == 'lc':
            self.dgc_r = LocallyConnectedGC(self.k)#, depthwise=False
            self.dgc_r.build(shapes)
            wr_1 = self.dgc_r.trainable_weights

            self.lin_r = Linear(1, activation=self.recurrent_activation)
            self.lin_r.build(shapes)
            wr_2 = self.lin_r.trainable_weights

            self.dgc_u = LocallyConnectedGC(self.k)#, depthwise=False
            self.dgc_u.build(shapes)
            wu_1 = self.dgc_u.trainable_weights

            self.lin_u = Linear(1, activation=self.recurrent_activation)
            self.lin_u.build(shapes)
            wu_2 = self.lin_u.trainable_weights

            self.dgc_c = LocallyConnectedGC(1)#, depthwise=False
            self.dgc_c.build(shapes)
            wc_1 = self.dgc_c.trainable_weights

            self.lin_c = Linear(1, activation=self.activation)
            self.lin_c.build(shapes)
            wc_2 = self.lin_c.trainable_weights
            
            w = wr_1+wr_2+wu_1+wu_2+wc_1+wc_2
        
        
        self._trainable_weights = w
        self.built = True

    def call(self, inputs, states):
        
        feature_ru = K.concatenate([K.expand_dims(inputs), K.expand_dims(states[0])]) #(bs, nb_nodes, 2)
        
        r = self.dgc_r(feature_ru) #(bs, nb_nodes, 2)
        if self.dgc_mode != 'dgc':
            r = self.lin_r(r) #(bs, nb_nodes, 1)
        
        u = self.dgc_u(feature_ru) #(bs, nb_nodes, 2)
        if self.dgc_mode != 'dgc':
            u = self.lin_u(u) #(bs, nb_nodes, 1)
        
        s = r*K.expand_dims(states[0]) #(bs, nb_nodes, 1)
        feature_c = K.concatenate([K.expand_dims(inputs), s]) #(bs, nb_nodes, 2)
        
        c = self.dgc_c(feature_c) #(bs, nb_nodes, 2)
        if self.dgc_mode != 'dgc':
            c = self.lin_c(c) #(bs, nb_nodes, 1)
        
        H = u*K.expand_dims(states[0]) + (1-u)*c
        outputs = K.squeeze(H, axis=-1)
        return outputs, [outputs]

##################################
class RecurrentDGC(Layer):
    """
    Dynamic Graph Filter Networks (DGFN) cell
    Eq-(DGFN) in the paper
        Arguments:
            k: receptive field
            units: output dimension
            scaled_lap: scaled Laplacian matrix
    """
    
    def __init__(self, 
                 k, 
                 units,
                 scaled_lap = scaled_laplacian,
                 inner_act=None,
                 activation='sigmoid',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RecurrentDGC,self).__init__(**kwargs)
        self.k = k
        self.units = units
        self.state_size = dim
        self.output_size = dim
        self.scaled_lap = scaled_lap
        self.inner_act=inner_act
        self.activation = activations.get(activation)
        self.supports_masking = True
        self.supports = []
        s = K.constant(K.to_dense(calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(s)
    
    def build(self, input_shape):
        #input_shape = (bs, nb_nodes)
        shapes = (input_shape[0], input_shape[1], 2)
        shapes_1 = (input_shape[0], input_shape[1], 1) #CAREFUL
        shapes_i = (input_shape[0], input_shape[1], self.units)

        self.dgc_r = DynamicGC(k=self.k, units=1, normalization=True, activation = None)
        self.dgc_r.build(shapes)
        w_1 = self.dgc_r.trainable_weights
        
        self.r1 = Linear(self.units, activation = self.activation)
        self.r1.build(shapes_1)
        w_2 = self.r1.trainable_weights
        
        self.r2 = Linear(1, activation = self.activation)
        self.r2.build(shapes_i)
        w_3 = self.r2.trainable_weights
            
        self._trainable_weights = w_1  + w_2 + w_3
        self.built = True

    def call(self, inputs, states):
        feature_ru = K.concatenate([K.expand_dims(inputs), K.expand_dims(states[0])])
        r = self.dgc_r(feature_ru) #(bs, nb_nodes, 2)
        
        #read = self.r1(r)
        #outputs = self.r2(read)
        #return K.squeeze(r,axis=-1), [K.squeeze(outputs,axis=-1)]
        
        # for model interpretation
        p = r[...,:1]
        read = self.r1(p)
        outputs = self.r2(read)
        return r, [K.squeeze(outputs,axis=-1)]
################################
class ScheduledSampling(Callback):
    '''
    Custom Callbacks to use teacher forcing and schedule sampling
        strategy in training.
        see `https://arxiv.org/pdf/1506.03099.pdf`
    # Argument:
        k: Interger, the coefficient to control convergence speed
        model: the keras model to apply
    '''
    
    def __init__(self, k, **kwargs):
        self.k = k
        super(ScheduledSampling, self).__init__(**kwargs)
    '''    
    def on_train_batch_begin(self, batch, logs=None):
        prob_use_gt = self.k / (self.k+math.exp((batch)/self.k))
        for layer in self.model.layers:
            if ('reset_coin' in dir(layer)):
                layer.reset_coin(prob_use_gt)
    '''
    def on_train_epoch_begin(self, epoch, logs=None):
        prob_use_gt = self.k / (self.k+math.exp(epoch/self.k))
        for layer in self.model.layers:
            if ('reset_coin' in dir(layer)):
                layer.reset_coin(prob_use_gt)
         
    def on_test_batch_begin(self, batch, logs=None):
        for layer in self.model.layers:
            if ('reset_coin' in dir(layer)):
                layer.reset_coin(0.)
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if ('reset_coin' in dir(layer)):
                layer.reset_coin(0.)