"""
Created 2018
Custom Layers include
- Factorization Machine Layer, building on initial code here: https://github.com/keras-team/keras/issues/4959 
- Mixture of Experts Layer, wrapping tensor2tensor.utils.expert_utils.local_moe
- Slicing layer to facilitate providing data to keras as a data frame or numpy array
"""
import tensorflow as tf
import keras
from keras.layers import Layer
from keras.engine.topology import InputSpec
from keras import backend as K
from keras import initializers, regularizers, activations, constraints
#from keras.layers.core import *

import tensor2tensor
from tensor2tensor.utils.expert_utils import local_moe
from tensor2tensor.utils.expert_utils import *

import numpy as np

class FactorizationMachinesLayer(Layer):
    '''Factorization Machines layer.

    # Arguments
        output_dim: int > 0.
        k: k of Factorization Machines
        init: name of initialization function for the weights of the layer
            (see [initializers](../initializers.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    '''
    def __init__(self,
                 output_dim, 
                 k                   =2,
                 input_dim           =None,
                 activation          =None,
                 bias                =True,
                 init                ='glorot_normal',
                 weights             = None,
                 W_regularizer       = None,
                 b_regularizer       = None,
                 activity_regularizer= None,
                 W_constraint        = None,
                 b_constraint        = None,
                 name                = None, 
                 sparse_inputs       = True,
                 interactions_3way   = False,
                 **kwargs):

        # print("Type test: ",type(self))
        
        if input_dim is not None:
            kwargs['input_shape'] = (input_dim,)
            # print ("kwargs dictionary test:", kwargs)

        assert isinstance(self, Layer) 
        # super(Layer,self).__init__(**kwargs)
        # print("Super Test:", super())
        # Layer.__init__(self,**kwargs)
        super().__init__(**kwargs)
        
        self.init                 = initializers.get(init)
        self.activation           = activations.get(activation)
        
        self.output_dim           = output_dim
        self.input_dim            = input_dim
        self.k                    = k

        self.W_regularizer        = regularizers.get(W_regularizer)
        self.b_regularizer        = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint         = constraints.get(W_constraint)
        self.b_constraint         = constraints.get(b_constraint)

        self.sparse_inputs        = sparse_inputs
        self.bias                 = bias
        self.initial_weights      = weights
        self.input_spec           = [InputSpec(min_ndim=2)]
        if not name:
            prefix = 'Factorization_Machine'
            name = prefix + '_' + str(K.get_uid(prefix))
        
        self.name                 = name 
        self.supports_masking     = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim                = input_shape[-1]
        self.input_dim           = input_dim
        self.input_spec          = [InputSpec(dtype=K.floatx(), min_ndim=2)]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer =self.init,
                                 name        ='{}_W'.format(self.name),
                                 regularizer =self.W_regularizer,
                                 constraint  =self.W_constraint)
        
        if self.sparse_inputs == True:
            # stick with 2 dimensional matrix multiplication which 
            # natively allows sparse inputs 
            V_dimensions_ = (input_dim, self.output_dim * self.k)
        else:
            # 3d tensor allows for simpler legacy code
            V_dimensions_ = (self.output_dim, input_dim, self.k)
            
        self.V = self.add_weight(V_dimensions_,
                                 initializer = self.init,
                                 name        = '{}_V'.format(self.name),
                                 regularizer = self.W_regularizer,
                                 constraint  = self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer= 'zero',
                                     name       = '{}_b'.format(self.name),
                                     regularizer= self.b_regularizer,
                                     constraint = self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            
        self.built = True

    def call(self, x, mask=None):
        output  = K.dot(x, self.W)
        if self.bias:
            output += self.b        
        if self.sparse_inputs == False:
            # version that works only with dense tensors
            output += K.sum(K.square(K.dot(x, self.V)) - K.dot(K.square(x), K.square(self.V)), axis = 2)
        elif self.sparse_inputs == True:
            # work only with 2 d matrix multiplication
            # batch_size = K.int_shape(x)[0] 
            output += K.sum(
                    K.reshape((K.square(K.dot(x, self.V)) - K.dot(K.square(x), K.square(self.V))),
                              [-1, self.output_dim, self.k]),
                    axis = 2)/2
        
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim':           self.output_dim,
                  'init':                 self.init.get_config(),
                  'activation':           activations.serialize(self.activation),
                  'W_regularizer':        self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer':        self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint':         self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint':         self.b_constraint.get_config() if self.b_constraint else None,
                  'bias':                 self.bias,
                  'input_dim':            self.input_dim}

        assert isinstance(self, Layer)
        # assert isinstance(self, FactorizationMachinesLayer)
        # base_config = super(FactorizationMachinesLayer, self).get_config()
        base_config = super().get_config()
        # base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))
    
# Local implementation of Mixture of Experts as a layer, wrapping functionality of local_moe operation
# Could take as inputs multiple expert models, either trained or not
class MixtureOfExpertsLayer(Layer):
    def __init__(self, 
                 list_of_expert_models,                  # list of layer objects - could use LayerWrapper for definitions
                 use_top_K                        = None,# number of experts to pay attention to for each record
                 output_dim                       = 1,   # output dimensions
                 input_dim                        = None,# provide input dimensions, if known, or None
                 coeff_loss_promoting_development = 1e-2,# coefficient to help promote uniform usage of experts
                 #sparse_inputs                    = False, # considering to add native support for this
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # additional initialization for MOE layer
        # save provided parameters
        self.list_of_expert_models            = list_of_expert_models
        self.units                            = len(list_of_expert_models)
        self.use_top_K                        = use_top_K or self.units
        self.output_dim                       = output_dim
        self.input_dim                        = input_dim
        self.coeff_loss_promoting_development = coeff_loss_promoting_development 
        #self.sparse_inputs                    = sparse_inputs
        
        create_callables = np.vectorize(lambda func: (lambda x:func(x)))
        self.list_of_expert_callables         = create_callables(list_of_expert_models).tolist()
        
        self.uses_learning_phase              = True
        #self.stateful                         = True
#        self.experts_initialized     = False

    def build(self, input_shape):
        #super().build(input_shape)
        
        for expert in self.list_of_expert_models:
            for weight in expert.trainable_weights:
                self.trainable_weights.append(weight)
            for weight in expert.non_trainable_weights:                
                self.non_trainable_weights.append(weight)
            
        self.built = True

#    def initialize_experts_if_needed(self,input_shape):
#        #initialize models if not yet built
#        K.get_session()
#        self.experts_initialized = True

    def call(self, inputs):
        # This function links together computational graph generating output of a MoE layer
        # compute assessments of relevant experts, and associated weights
        
        #if (self.sparse_inputs == True) and (self.input_dim is not None):
        if K.is_sparse(inputs) and (self.input_dim is not None):            # - alternative pre-condition
            raise ValueError('Note support for sparse inputs is not implemented yet')            
            # TODO (PG) add sparse support to prescribe shape of the sparse input tensor and allow shape inference
        
        weighted_sum, extra_training_loss = local_moe(        
                inputs,
                K.learning_phase(), 
                self.list_of_expert_callables,
                self.units,
                self.use_top_K,
                loss_coef=self.coeff_loss_promoting_development)
        
        #self.losses.append(extra_training_loss)
        self.add_loss(extra_training_loss, inputs=inputs)

        return weighted_sum


    def compute_output_shape(self, input_shape):        
        return (input_shape[0:-1], self.output_dim)


    def get_config(self):
        # dictionary with all elements necessary for init to re-construct this object
        config = {'list_of_expert_models'           : self.list_of_expert_models,
                  'use_top_K'                       : self.use_top_K,
                  'output_dim'                      : self.output_dim,
                  'coeff_loss_promoting_development': self.coeff_loss_promoting_development}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def SingleSliceLayer2d(column_name, column_index):
    return keras.layers.Lambda(lambda tensor: K.expand_dims(tensor[:,column_index]), output_shape=(1,), name="pick_"+ column_name)

#note this layer assumes the input data is consistently structured over multiple feedings into the model
def MultipleSliceLayers2d(column_names, typical_input_data_frame = None): 
    column_index_lookup = lambda name: typical_input_data_frame.columns.tolist().index(name)
    if type(column_names) is not list: column_names = [column_names]
    return [SingleSliceLayer2d(column_name, column_index_lookup(column_name)) for column_name in column_names]
