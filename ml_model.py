import tensorflow as tf
import numpy as np
import graph_nets as gn
import sonnet as snt

from helper_functions import *

def mean_weighted_error(y_true, y_pred):
    x = (y_pred - y_true)/tf.norm(y_true, axis = -1, keepdims=True)
    x = tf.norm(x, axis=-1)
    loss = tf.reduce_mean(x)
    return loss

class MeanWeightedError(tf.keras.metrics.Metric):
    def __init__(self, name="mean_weighted_error", **kwargs):
        super(MeanWeightedError, self).__init__(name=name, **kwargs)
        self.mwe = self.add_weight(name="mwe", initializer="zeros")

    def update_state(self, y_true, y_pred):
        self.mwe.assign_add(mean_weighted_error(y_true, y_pred))

    def result(self):
        return self.mwe

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mwe.assign(0.0)

loss_tracker = tf.keras.metrics.Mean(name='loss')
loss_test = MeanWeightedError(name='loss_test')

class LearnForces(tf.keras.Model):
    def __init__(self, nplanets, senders, receivers, noise_level = 0.):
        super(LearnForces, self).__init__()
        self.noise_level = noise_level
        self.senders = senders
        self.receivers = receivers
        self.nplanets = nplanets
        self.nedges = nplanets*(nplanets-1)//2

        self.opt1 = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.opt2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
        #self.test_loss_metric = tf.keras.metrics.MeanAbsoluteError(name='test_loss')
        
        logm_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        #logm_init = tf.constant_initializer(np.log10(masses[1:]))
        #logG_init = tf.constant_initializer(np.log10(G/A_norm))
        logG_init = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        
        M = tf.constant_initializer([
                         [-2, 0., 0.], 
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.], 
                         [1., 0., 0.], 
                         [1., 0., 0.],
                                    ],)
        
        self.logm_planets = tf.Variable(
            initial_value=logm_init(shape=(self.nplanets-1,), dtype="float32"),
            trainable=True,
        )
        #a = tf.constant([0.], dtype = tf.float32)
        #self.logm = tf.concat([a, self.logm_planets], axis=0)
        
        self.logG = tf.Variable(
            initial_value=logG_init(shape=(1,), dtype="float32"),
            trainable=True,
        )
        
        self.graph_network = gn.blocks.EdgeBlock(
            #edge_model_fn=lambda: snt.Linear(3, with_bias = False, 
            #                                 w_init=M),
            #edge_model_fn=lambda: snt.nets.MLP([32, 32, 3],
            #                                  with_bias = True,
            #                                  activation = tf.keras.activations.tanh),
            edge_model_fn = lambda: snt.Sequential([
                                                  #normalizer,
                                                  tf.keras.layers.Dense(128, input_dim=6, kernel_initializer='normal', activation='relu'),
                                                  tf.keras.layers.Dense(128, activation='relu'),
                                                  tf.keras.layers.Dense(128, activation='relu'),
                                                  #tf.keras.layers.Dense(32, input_dim=6, kernel_initializer='normal', activation='relu'),
                                                  #tf.keras.layers.Dense(32, activation='relu'),
                                                  #tf.keras.layers.Dense(32, activation='relu'),
                                                  tf.nn.relu,
                                                  snt.Linear(3),
                                                            ]),
            use_edges = True,
            use_receiver_nodes = True,
            use_sender_nodes = True,
            use_globals = True,
        )
                


    def sum_forces(self, graph):
        b1_tr = gn.blocks.ReceivedEdgesToNodesAggregator(reducer = tf.math.unsorted_segment_sum)(graph)
        b2_tr = gn.blocks.SentEdgesToNodesAggregator(reducer = tf.math.unsorted_segment_sum)(graph)
        summed_forces = b1_tr-b2_tr
        return summed_forces
            
    def get_acceleration(self, forces, graph):
        acceleration_tr = tf.divide(forces, tf.pow(10.,graph.nodes))
        return acceleration_tr
        #output_ops_tr = tf.reshape(acceleration_tr, shape=[self.ntime, self.nplanets, 3])
        #return output_ops_tr
        
    def call(self, D, training = False, extract = False):
        #self.ntime = len(g.nodes)//nplanets
        ntime = len(D)//self.nedges
        if training == True:
            m_noise = tf.random.normal(tf.shape(self.logm_planets), 0, self.noise_level, tf.float32)
            lm = self.logm_planets*(1+ m_noise)
        else: 
            lm = self.logm_planets
            
        a = tf.constant([np.log10(5.522376708530351)], dtype = tf.float32)
        lm = tf.concat([a, lm], axis=0)
        
        #nodes_g = np.concatenate([np.log10(masses_tf)]*ntime)[:,np.newaxis]
        nodes_g = tf.concat([lm]*ntime, axis = 0)
        nodes_g = tf.expand_dims(nodes_g, 1)
        senders_g, receivers_g = reshape_senders_receivers(self.senders, 
                                                             self.receivers, 
                                                             ntime, 
                                                             self.nplanets, 
                                                             self.nedges)
        
        # Create graph
        graph_dict = { 
          "nodes": nodes_g,
          "edges": cartesian_to_spherical_coordinates(D), 
          "receivers": receivers_g, 
          "senders": senders_g ,
          "globals": self.logG
           } 
    
        g = gn.utils_tf.data_dicts_to_graphs_tuple([graph_dict])

        g = self.graph_network(g)
        g = g.replace(
            edges = spherical_to_cartesian_coordinates(g.edges))
        f = self.sum_forces(g)
        a = self.get_acceleration(f, g)
        if extract == True: 
            f = tf.reshape(g.edges, shape=[-1, self.nedges, 3]).numpy()
            a = tf.reshape(a, shape=[-1, self.nplanets, 3]).numpy()
            return a, f
        else: 
            return a
    
    def train_step(self, data):
        #if isinstance(data, tuple):
        #    data = data[0]
        # Unpack the data
        D, A = data
        
        D_rot, A_rot = rotate_data(D, A)
        #D_rot, A_rot = D, A
        
        D_noise = tf.random.normal(tf.shape(D), 0, self.noise_level, tf.float32)
        D_rot = D_rot*(1+ D_noise)

        # Randomly 3D rotate the data
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(D_rot, training = True)
            # Compute the loss
            loss = mean_weighted_error(A_rot, predictions)
        
        # Compute gradients
        # Trainable variables are the masses and the MLP layers 
        #Trainable_vars = self.trainable_variables+ list(self.graph_network.trainable_variables)
        #gradients = tape.gradient(loss, trainable_vars)
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # Update weights
        #self.optimizer.apply_gradients(zip(gradients,trainable_vars))


        var_list1 = self.trainable_variables
        var_list2 = list(self.graph_network.trainable_variables)
        gradients = tape.gradient(loss, var_list1 + var_list2)
        grads1 = gradients[:len(var_list1)]
        grads2 = gradients[len(var_list1):]
        #grads1, _ = tf.clip_by_global_norm(grads1, 5.0)
        #grads2, _ = tf.clip_by_global_norm(grads2, 5.0)
        train_op1 = self.opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = self.opt2.apply_gradients(zip(grads2, var_list2)) 
        train_op = tf.group(train_op1, train_op2)        
        
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    def test_step(self,data):
        # Unpack the data
        D, A = data
        
        predictions = self(D)

        loss_test.update_state(A, predictions)
        
        return {"loss": loss_test.result()}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, loss_test]
