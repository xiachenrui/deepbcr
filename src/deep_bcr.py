##########################################################################################
# Deep learning models for BCR repertoires
#
# Author: Xihao Hu <huxihao@gmail.com>
#
# @MIT License
##########################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tensorflow as tf

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'

class RepertoireModel:
    """ High level class for common repertoire analysis.

            train():        train a deep learning model
            bacth_train():  train a deep learning model in batch
            test():         test the model performance
            predict():      predict a new dataset with unknown labels
        
    """

    def __init__(self, model_name='RepertoireModel', save_path=''):
        self.model_name = model_name
        self.save_path = save_path

    def variable_summary(self, var, name=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if name is None:
           name = var.op.name.split('/')[-1]+'-summary'
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def logits(self, features, counts, keep_prob):
        """ Get the logits from predictions """
        pass

    def objective(self, logits, labels):
        """ Get the objective function """
        with tf.name_scope('objective'):
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
#            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar('summary', cross_entropy)
            return cross_entropy

    def accuracy(self, logits, labels):
        """ Get the prediction accuracy, sensitivity and specificity """
        with tf.name_scope('accuracy'):
            probabilities = tf.sigmoid(logits)
            correct = tf.equal(tf.round(labels), tf.round(probabilities))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), axis=0)
            pos = tf.not_equal(tf.round(labels), 0)
            neg = tf.equal(tf.round(labels), 0)
            sensitivity = tf.reduce_sum(tf.cast(tf.logical_and(correct, pos), tf.float32), axis=0) / \
                          tf.reduce_sum(tf.cast(pos, tf.float32), axis=0)
            specificity = tf.reduce_sum(tf.cast(tf.logical_and(correct, neg), tf.float32), axis=0) / \
                          tf.reduce_sum(tf.cast(neg, tf.float32), axis=0)
            tf.summary.scalar('summary', accuracy)
            tf.summary.scalar('summary', sensitivity)
            tf.summary.scalar('summary', specificity)
            return accuracy, sensitivity, specificity

    def performance(self, out):
        print(self.model_name, 'Objective:', '%4.3f'%out[1], end=' ')
        print('Acc:', '%4.1f'%(100.0*out[2][0]), end=' ')
        print('Sen:', '%4.1f'%(100.0*out[2][1]), end=' ')
        print('Spe:', '%4.1f'%(100.0*out[2][2]))
        return out[1], out[2]

    def pre_train(self, xs, cs, ys):
        """ Function for initialization """
        self.save_list = []

    def get_update_list(self):
        """ Get the list of parameters for updating """
        if hasattr(self, 'update_list'):
            return self.update_list
        else:
            return self.save_list

    def post_train(self, xs, cs, ys):
        """ Function for model selection """
        pass

    def update_model_path(self, select=None, ext='.index'):
        path = os.path.join(self.save_path, self.model_name+'-par/')
        if not os.path.exists(path):
            return None
        ifile = []
        for f in os.listdir(path):
            if f.startswith('model') and f.endswith(ext):
                m = os.path.join(path, f[:-len(ext)])
                i = int(m.split('-')[-1])
                if select is None or i <= select:
                    ifile.append((i, m))
        if len(ifile) > 0:
            ifile.sort(reverse=True)
            self.last_saved_model = os.path.abspath(ifile[0][1])
            return ifile[0][0]
        return None

    def encode_labels(self, ys):
        return ys

    def train(self, xs, cs, ys, 
              dropout_keep_prob=0.5, 
              learning_rate=0.001, 
              max_iterations=1000, 
              it_step_size=500,
              recover=True,
              save_log=False):

        """ General function for training a deep learning model
            
            Args:
                xs: data matrix of shape (#samples, #feature1, #feature2 ...)
                cs: indicating matrix for valid values in xs, shape (#samples, #feature1)
                ys: labels (starting from 0) for learning, shape (#samples)
                dropout_keep_prob: default dropout probability in learning
                learning_rate: the parameter in Adam optimizer
                max_iterations: maximum number of iterations to perform
                it_step_size: number of iterations for saving learnt models

            Returns:
                (logits, objective, accuracy, merged_log, optimizer)

            Features:
                * save intermedial steps
                * recover from a saved model

        """
        ys = self.encode_labels(ys)
        tf.reset_default_graph() ## clean up
        tf.set_random_seed(0)

        self.pre_train(xs, cs, ys)

        with tf.name_scope('inputs'):
            features = tf.placeholder(tf.float32, xs.shape)
            counts = tf.placeholder(tf.float32, cs.shape)
            labels = tf.placeholder(tf.float32, ys.shape)
            keep_prob = tf.placeholder(tf.float32)

        logits = self.logits(features, counts, keep_prob)
        objective = self.objective(logits, labels)
        accuracy = self.accuracy(logits, labels)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(objective, var_list=self.get_update_list())

        merged_log = tf.summary.merge_all()

        saver = tf.train.Saver(self.save_list, max_to_keep=1)

        self.update_model_path(max_iterations)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            if recover and hasattr(self, 'last_saved_model') and os.path.exists(self.last_saved_model+'.index'):
                print('Load trained', self.model_name, 'from', self.last_saved_model)
                saver.restore(session, save_path=self.last_saved_model)
                start = int(self.last_saved_model.split('-')[-1])
            else:
                start = 0
            if save_log:
                log_writer = tf.summary.FileWriter(os.path.join(self.save_path, self.model_name+'-log'), session.graph)
            if start > max_iterations:
                raise AssertionError('Please clean up '+self.last_saved_model)
            out = session.run(
                (logits, objective, accuracy, merged_log),
                feed_dict={features: xs, counts: cs, labels: ys, keep_prob: dropout_keep_prob}
            )
            print('Before run', end=':\t')
            self.performance(out)
            for iteration in range(start+1, max_iterations+1):
                out = session.run(
                    (logits, objective, accuracy, merged_log, optimizer),
                    feed_dict={features: xs, counts: cs, labels: ys, keep_prob: dropout_keep_prob}
                )
                if save_log and iteration % 10 == 0:
                    log_writer.add_summary(out[3], iteration)
                if iteration % it_step_size == 0 or iteration == max_iterations:
                    print('iter', iteration, end=':\t')
                    self.performance(out)
                    if iteration == max_iterations: ## only save the final model
                        f = saver.save(session, save_path=os.path.join(self.save_path, self.model_name+'-par/model'), 
                                global_step=iteration, write_meta_graph=False)
                        self.last_saved_model = f
        self.post_train(xs, cs, ys) ## in case we need to run something just after the training
        return out

    def load(self, xs, cs, ys, max_iterations=None):
        ys = self.encode_labels(ys)
        tf.reset_default_graph() ## clean up
        tf.set_random_seed(0)
        self.last_saved_model = None ## clean up

        self.pre_train(xs, cs, ys)
        i = self.update_model_path(max_iterations)
        if i is None or (max_iterations is not None and i != max_iterations):
            raise RuntimeError('No saved models to load in '+self.save_path+'\nPlease train the model again!')
        self.post_train(xs, cs, ys) ## in case we need to run something just after the training

    def test(self, xs, cs, ys):
        """ Predict and calculate accuracy """
        ys = self.encode_labels(ys)

        features = tf.placeholder(tf.float32, xs.shape)
        counts = tf.placeholder(tf.float32, cs.shape)
        labels = tf.placeholder(tf.float32, ys.shape)
        keep_prob = tf.placeholder(tf.float32)

        logits = self.logits(features, counts, keep_prob)
        objective = self.objective(logits, labels)
        accuracy = self.accuracy(logits, labels)

        with tf.Session() as session:
#            print('Load trained', self.model_name, 'from', self.last_saved_model)
            saver = tf.train.Saver(self.save_list)
            saver.restore(session, save_path=self.last_saved_model)
            out = session.run(
                (logits, objective, accuracy),
                feed_dict={features: xs, counts: cs, labels: ys, keep_prob: 1.0}
            )
            self.performance(out)
        return out

    def predict(self, xs, cs):
        """ Only for prediction """
        features = tf.placeholder(tf.float32, xs.shape)
        counts = tf.placeholder(tf.float32, cs.shape)
        keep_prob = tf.placeholder(tf.float32)

        logits = self.logits(features, counts, keep_prob)

        with tf.Session() as session:
#            print('Load trained', self.model_name, 'from', self.last_saved_model)
            saver = tf.train.Saver(self.save_list)
            saver.restore(session, save_path=self.last_saved_model)
            out = session.run(logits, feed_dict={features: xs, counts: cs, keep_prob: 1.0})
        return out

    def train_batch(self, data_iter,
                    dropout_keep_prob=0.5, 
                    learning_rate=0.01, 
                    max_iterations=100, 
                    it_step_size=20,
                    rate_decay_factor=1.002):

        """ General function for training a deep learning model in batches
            
            Args:
                data_iter: data iterater
                learning_rate: the parameter in Adam optimizer
                max_iterations: maximum number of iterations to perform
                it_step_size: number of iterations for saving learned models

            Remarks:
                We set a learning rate decay as:
                    learning_rate / rate_decay_factor ** max_iterations

            Returns:
                a list of (iteration, objective, accuracy)
        """
        tf.reset_default_graph() ## clean up
        tf.set_random_seed(0)

        xs, cs, ys = next(data_iter)
        ys = self.encode_labels(ys)
        self.pre_train(xs, cs, ys)

        with tf.name_scope('inputs'):
            features = tf.placeholder(tf.float32, xs.shape)
            counts = tf.placeholder(tf.float32, cs.shape)
            labels = tf.placeholder(tf.float32, ys.shape)
            keep_prob = tf.placeholder(tf.float32)

        logits = self.logits(features, counts, keep_prob)
        objective = self.objective(logits, labels)
        accuracy = self.accuracy(logits, labels)

        with tf.name_scope('train'):
            lr = learning_rate  / rate_decay_factor ** max_iterations
            print('Learning rate is', lr)
            optimizer = tf.train.AdamOptimizer(lr).minimize(objective, var_list=self.get_update_list())

        saver = tf.train.Saver(self.save_list, max_to_keep=1)

        self.update_model_path(max_iterations)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            if hasattr(self, 'last_saved_model') and os.path.exists(self.last_saved_model+'.index'):
                print('Load trained', self.model_name, 'from', self.last_saved_model)
                saver.restore(session, save_path=self.last_saved_model)
                start = int(self.last_saved_model.split('-')[-1])
            else:
                start = 0
            if start > max_iterations:
                raise AssertionError('Please clean up '+self.last_saved_model)
            out = session.run(
                (logits, objective, accuracy),
                feed_dict={features: xs, counts: cs, labels: ys, keep_prob: dropout_keep_prob}
            )
            print('Before run', end=':\t')
            self.performance(out)
            for iteration in range(start+1, max_iterations+1):
                out = session.run(
                    (logits, objective, accuracy, optimizer),
                    feed_dict={features: xs, counts: cs, labels: ys, keep_prob: dropout_keep_prob}
                )
                if iteration % it_step_size == 0 or iteration == max_iterations:
                    print('iter', iteration, end=':\t')
                    self.performance(out)
                    if iteration == max_iterations: ## only save the final model
                        f = saver.save(session, save_path=os.path.join(self.save_path, self.model_name+'-par/model'), 
                                global_step=iteration, write_meta_graph=False)
                        self.last_saved_model = f

                    ## feed new data
                    xs, cs, ys = next(data_iter)
                    ys = self.encode_labels(ys)

        self.post_train(xs, cs, ys) ## in case we need to run something just after the training
        return out

class MaxSnippetModel(RepertoireModel):
    """ Logistic regression model for BCR repertoires

        Modified from the MaxSnippetModel class in
        https://github.com/jostmey/MaxSnippetModel
    """

    def __init__(self, num_replicas=100, name='MaxSnippetModel', save_path=''):
        self.model_name = name
        self.save_path = save_path
        self.num_replicas = num_replicas

    def performance(self, out):
        if self.bestfit_index is None: ## for training
            i = np.argmin(out[1])
        else: ## for testing
            i = self.bestfit_index
        print(self.model_name, 'Objective:', '%4.3f'%out[1][i], 'Accuracy:', '%4.3f'%(100.0*out[2][i]))
        return out[1][i], out[2][i]

    def pre_train(self, xs, cs, ys):
        with tf.variable_scope('parameters'):
            self.weights = tf.get_variable('W', [xs.shape[2], self.num_replicas],
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases = tf.get_variable('b', [self.num_replicas], initializer=tf.constant_initializer(0.0))
        self.save_list = [self.weights, self.biases]
        self.bestfit_index = None

    def post_train(self, xs, cs, ys):
        features = tf.placeholder(tf.float32, xs.shape)
        counts = tf.placeholder(tf.float32, cs.shape)
        labels = tf.placeholder(tf.float32, ys.shape) 
        keep_prob = tf.placeholder(tf.float32)

        logits = self.logits(features, counts)
        objective = self.objective(logits, labels)

        with tf.name_scope('post_train'):
            index_bestfit = tf.argmin(objective, 0)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            if self.save_path is not None:
                saver = tf.train.Saver(self.save_list)
                saver.restore(session, save_path=self.last_saved_model)
            out = session.run(
                (index_bestfit, self.weights, self.biases),
                feed_dict={features: xs, counts: cs, labels: ys, keep_prob: 1.0}
            )
            print('Best model is at index', out[0])
            self.bestfit_index = out[0]

    def logits(self, features, counts, keep_prob=None):
        shape = features.get_shape()
        batch_size = int(shape[0])
        max_instances = int(shape[1])
        num_features = int(shape[2])
        with tf.variable_scope('logit'):
            features_flat = tf.reshape(features, [batch_size*max_instances, num_features])
            scores_flat = tf.matmul(features_flat, self.weights)+self.biases
            scores = tf.reshape(scores_flat, [batch_size, max_instances, self.num_replicas])
            counts_expand = tf.expand_dims(counts, axis=2)

            penalties = -1E12*(1.0-tf.sign(counts_expand))  
            # No penalty if counts > 0. The penalty is -1E12 when counts are zero.

            tf.summary.histogram('logits', scores_flat)
            return tf.reduce_max(scores+penalties, axis=1)

    def objective(self, logits, labels):
        with tf.name_scope('objective') as scope:
            labels_expand = tf.expand_dims(labels, axis=1)
            labels_tile = tf.tile(labels_expand, [1, self.num_replicas])
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_tile), axis=0)
            self.variable_summary(cross_entropy, 'summary')
            return cross_entropy

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            probabilities = tf.sigmoid(logits)
            labels_expand = tf.expand_dims(labels, axis=1)
            labels_tile = tf.tile(labels_expand, [1, self.num_replicas])
            correct = tf.equal(tf.round(labels_tile), tf.round(probabilities))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), axis=0)
            self.variable_summary(accuracy, 'summary')
            return accuracy

class TwoLayerModel(RepertoireModel):
    """ A simple two-layer nerual network """

    def __init__(self, num_motifs=100, model_name='TwoLayerModel', save_path=''):
        self.model_name = model_name
        self.num_motifs = num_motifs
        self.save_path = save_path

    def pre_train(self, xs, cs, ys):
        with tf.variable_scope('parameters'):
            self.weights1 = tf.get_variable('W1', [xs.shape[2], self.num_motifs],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases1 = tf.get_variable('b1', [self.num_motifs], initializer=tf.constant_initializer(0.1))
            self.weights2 = tf.get_variable('W2', [self.num_motifs, 1],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.1))
        self.save_list = [self.weights1, self.biases1, self.weights2, self.biases2]

    def logits(self, features, counts, keep_prob):
        shape = features.get_shape()
        batch_size = int(shape[0])
        max_instances = int(shape[1])
        num_features = int(shape[2])

        with tf.name_scope('layer1'):
            features_flat = tf.reshape(features, [batch_size*max_instances, num_features])
            scores_flat = tf.matmul(features_flat, self.weights1) + self.biases1
            tf.summary.histogram('logits', scores_flat)
            scores = tf.reshape(scores_flat, [batch_size, max_instances, self.num_motifs])
            counts_expand = tf.expand_dims(counts, axis=2)
            valid = tf.multiply(scores, tf.sign(counts_expand))
            logits1 = tf.reduce_max(valid, axis=1)
        with tf.name_scope('layer2'):
            logits2 = tf.squeeze(tf.matmul(tf.nn.relu(logits1), self.weights2) + self.biases2)
            tf.summary.histogram('logits', scores_flat)
        return logits2

class MultipleLabelModel(RepertoireModel):
    """ Two-layer nerual network for multiple label outputs """

    def __init__(self, num_motifs=100, num_labels=2, model_name='MultipleLabelModel', save_path=''):
        self.model_name = model_name
        self.num_motifs = num_motifs
        self.num_labels = num_labels
        self.save_path = save_path

    def pre_train(self, xs, cs, ys):
        with tf.variable_scope('parameters'):
            self.weights1 = tf.get_variable('W1', [xs.shape[2], self.num_motifs],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases1 = tf.get_variable('b1', [self.num_motifs], initializer=tf.constant_initializer(0.1))
            self.weights2 = tf.get_variable('W2', [self.num_motifs, self.num_labels],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases2 = tf.get_variable('b2', [self.num_labels], initializer=tf.constant_initializer(0.1))
        self.save_list = [self.weights1, self.biases1, self.weights2, self.biases2]

    def encode_labels(self, ys):
        """ Expand the integer labels to be binary vectors """
        new = np.vstack([ys == i for i in range(self.num_labels)]).T
        return np.array(new, dtype=int)

    def logits(self, features, counts, keep_prob):
        shape = features.get_shape()
        batch_size = int(shape[0])
        max_instances = int(shape[1])
        num_features = int(shape[2])

        with tf.name_scope('layer1'):
            features_flat = tf.reshape(features, [batch_size*max_instances, num_features])
            scores_flat = tf.matmul(features_flat, self.weights1) + self.biases1
            tf.summary.histogram('logits', scores_flat)
            scores = tf.reshape(scores_flat, [batch_size, max_instances, self.num_motifs])
            counts_expand = tf.expand_dims(counts, axis=2)
            valid = tf.multiply(scores, tf.sign(counts_expand))
            logits1 = tf.reduce_max(valid, axis=1)
        with tf.name_scope('layer2'):
            logits2 = tf.squeeze(tf.matmul(tf.nn.relu(logits1), self.weights2) + self.biases2)
        return logits2

    def accuracy(self, logits, labels):
        """ Get the prediction accuracy, sensitivity and specificity """
        with tf.name_scope('accuracy'):
            y_pred = tf.nn.softmax(logits)
            y_pred_cls = tf.argmax(y_pred, axis=1)
            labels_cls = tf.argmax(labels, axis=1)
            correct = tf.equal(labels_cls, y_pred_cls)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('summary', accuracy)
            return accuracy

    def performance(self, out):
        print(self.model_name, 'Objective:', '%4.3f'%out[1], end=' ')
        print('Accuracy:', '%4.1f'%(100.0*out[2]))
        return out[1], out[2]

class EncodingLayerModel(MultipleLabelModel):
    """ Automatic update the weights for encoding amino acid sequences """

    def __init__(self, num_motifs=100, encode_init=None, num_labels=2, model_name='EncodingLayerModel', save_path=''):
        self.model_name = model_name
        self.num_motifs = num_motifs
        self.num_labels = num_labels
        self.save_path = save_path

        if encode_init is None:
            self.encode_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
            self.encode_shape = (len(AA_LIST), len(AA_LIST))
        elif type(encode_init) is int:
            self.encode_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
            self.encode_shape = (len(AA_LIST), encode_init)
        else:
            self.encode_init = tf.constant_initializer(encode_init)
            self.encode_shape = encode_init.shape
        print('Encoding matrix has shape', self.encode_shape)

    def pre_train(self, xs, cs, ys):
        batch_size, max_kmer, kmer_size = xs.shape
        amino_acid, encode_size = self.encode_shape
        with tf.variable_scope('parameters'):
            self.weights0 = tf.get_variable('W0', [amino_acid, encode_size], initializer=self.encode_init)

            self.weights1 = tf.get_variable('W1', [kmer_size*encode_size, self.num_motifs],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases1 = tf.get_variable('b1', [self.num_motifs], initializer=tf.constant_initializer(0.1))

            self.weights2 = tf.get_variable('W2', [self.num_motifs, self.num_labels],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases2 = tf.get_variable('b2', [self.num_labels], initializer=tf.constant_initializer(0.1))
        self.save_list = [self.weights0, self.weights1, self.biases1, self.weights2, self.biases2]

    def logits(self, features, counts, keep_prob):
        batch_size, max_kmer, kmer_size = features.get_shape()
        batch_size = int(batch_size)
        max_kmer = int(max_kmer)
        kmer_size = int(kmer_size)
        amino_acid, encode_size = self.encode_shape

        with tf.name_scope('layer0'):
            features_flat = tf.reshape(tf.to_int32(features), [-1])
            scores = tf.gather(self.weights0, features_flat)
            scores = tf.reshape(scores, [batch_size*max_kmer, kmer_size*encode_size])

        with tf.name_scope('layer1'):
            scores = tf.matmul(scores, self.weights1) + self.biases1
            scores = tf.reshape(scores, [batch_size, max_kmer, self.num_motifs])
            counts_expand = tf.expand_dims(counts, axis=2)
            valid = tf.multiply(scores, tf.sign(counts_expand))
            logits1 = tf.reduce_max(valid, axis=1)

        with tf.name_scope('layer2'):
            logits2 = tf.squeeze(tf.matmul(tf.nn.relu(logits1), self.weights2) + self.biases2)
        return logits2


class GeneSwitchModel(EncodingLayerModel):
    """ Add Ig constant gene switch events into the model """

    def __init__(self, num_motifs=30, encode_init=None, num_labels=2, model_name='GeneSwitchModel', save_path=''):
        self.model_name = model_name
        self.num_motifs = num_motifs
        self.num_labels = num_labels
        self.save_path = save_path

        if encode_init is None:
            self.encode_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
            self.encode_shape = (len(AA_LIST), len(AA_LIST))
        elif type(encode_init) is int:
            self.encode_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
            self.encode_shape = (len(AA_LIST), encode_init)
        else:
            self.encode_init = tf.constant_initializer(encode_init)
            self.encode_shape = encode_init.shape
        print('Encoding matrix has shape', self.encode_shape, 'num_motifs', num_motifs, 'num_labels', num_labels)

    def pre_train(self, xs, cs, ys):
        batch_size, max_kmer, kmer_size = xs.shape
        batch_size, max_kmer, gene_num = cs.shape
        amino_acid, encode_size = self.encode_shape
        with tf.variable_scope('parameters'):
            self.weights0 = tf.get_variable('W0', [amino_acid, encode_size], initializer=self.encode_init)

            self.weights1 = tf.get_variable('W1', [kmer_size*encode_size, self.num_motifs],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases1 = tf.get_variable('b1', [self.num_motifs], initializer=tf.constant_initializer(0.1))

            self.weights2 = tf.get_variable('W2', [self.num_motifs, gene_num],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases2 = tf.get_variable('b2', [self.num_motifs], initializer=tf.constant_initializer(0.1))

            self.weights3 = tf.get_variable('W3', [self.num_motifs, self.num_labels],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases3 = tf.get_variable('b3', [self.num_labels], initializer=tf.constant_initializer(0.1))
        self.save_list = [self.weights0, self.weights1, self.biases1, self.weights2, self.biases2, self.weights3, self.biases3]

    def hidden_layers(self, features, counts, keep_prob):
        batch_size, max_kmer, kmer_size = features.get_shape()
        batch_size, max_kmer, gene_num = counts.get_shape()
        amino_acid, encode_size = self.encode_shape

        batch_size = int(batch_size) ## B
        max_kmer = int(max_kmer)     ## M
        kmer_size = int(kmer_size)   ## K
        gene_num = int(gene_num)     ## C
        ## self.num_motifs          --> N
        ## self.num_labels          --> n
        ## amino_acid               --> A
        ## encode_size              --> E

        ## So, features is (B,M,K) and counts is (B,M,C)

        with tf.name_scope('encoding_layer'):
            features_flat = tf.reshape(tf.to_int32(features), [-1]) ## (B*M*K)
            scores = tf.gather(self.weights0, features_flat)
            ## the shape is (B*M*K,E)

        with tf.name_scope('kmer_layer'):
            scores = tf.reshape(scores, [batch_size*max_kmer, kmer_size*encode_size]) ## (B*M,K*E)
            scores = tf.matmul(scores, self.weights1) + self.biases1 ## (K*E,N)
            scores = tf.nn.relu(scores) ## kmers with non-positive scores will be ignored
            ## the shape is (B*M,N)

        with tf.name_scope('gene_layer'):
            scores = tf.reshape(scores, [batch_size*max_kmer, self.num_motifs]) ## (B*M,N)
            counts = tf.reshape(counts, [batch_size*max_kmer, gene_num]) ## (B*M,C)
            scores_expand = tf.expand_dims(scores, axis=2) ## (B*M,N,1)
            counts_expand = tf.expand_dims(counts, axis=1) ## (B*M,1,C)
            scores = tf.matmul(scores_expand, tf.sign(counts_expand)) ## ignore null kmers
            ## the shape is (B*M,N,C)

        with tf.name_scope('max_pooling'):
            scores = tf.reshape(scores, [batch_size, max_kmer, self.num_motifs, gene_num]) ## (B,M,N,C)
            scores = tf.reduce_max(scores, axis=1)
            ## the shape is (B,N,C)
#            scores = tf.nn.l2_normalize(scores, axis=0)

        with tf.name_scope('dropout_layer'):
            scores = tf.nn.dropout(scores, keep_prob)

        with tf.name_scope('motif_layer'):
            scores = scores * self.weights2 ## broadcasting to the first dimention
            scores = tf.reduce_sum(scores, axis=2) + self.biases2 ## (B,N)
            scores = tf.nn.relu(scores) ## select kmers with specific Ig isotypes
            ## the shape is (B,N)

        with tf.name_scope('dropout_layer'):
            scores = tf.nn.dropout(scores, keep_prob)

        return scores

    def logits(self, features, counts, keep_prob):
        scores = self.hidden_layers(features, counts, keep_prob)
        batch_size, max_kmer, kmer_size = features.get_shape()
        batch_size = int(batch_size) ## B

        with tf.name_scope('output_layer'):
            scores = tf.squeeze(tf.matmul(scores, self.weights3) + self.biases3) ## (N,n)
            scores = tf.reshape(scores, [batch_size, -1]) ## fix one sample case
            ## the shape is (B,n)
        return scores

class GeneSwitchModelFast(GeneSwitchModel):
    """ Fast computing the logits in the GeneSwitchModel """

    def hidden_layers(self, features, counts, keep_prob):
        batch_size, max_kmer, kmer_size = features.get_shape()
        batch_size, max_kmer, gene_num = counts.get_shape()
        amino_acid, encode_size = self.encode_shape

        batch_size = int(batch_size) ## B
        max_kmer = int(max_kmer)     ## M
        kmer_size = int(kmer_size)   ## K
        gene_num = int(gene_num)     ## C
        ## self.num_motifs          --> N
        ## self.num_labels          --> n
        ## amino_acid               --> A
        ## encode_size              --> E

        ## So, features is (B,M,K,A) and counts is (B,M,C)
        with tf.name_scope('pre-process'):
            counts = tf.reshape(counts, [batch_size*max_kmer, gene_num]) ## (B*M,C)
            valid = tf.reduce_sum(counts, axis=1) > 0
            scores = tf.reshape(tf.to_int32(features), [batch_size*max_kmer, kmer_size]) ## (B*M,K)
            scores = tf.boolean_mask(scores, valid) ## mask out zero inputs
            ## Now, the first dimension must be B*M or its products

        with tf.name_scope('encoding_layer'):
            scores = tf.reshape(scores, [-1]) ## (B*M*K)
            scores = tf.gather(self.weights0, scores)
            ## the shape is (B*M*K,E)

        with tf.name_scope('kmer_layer'):
            scores = tf.reshape(scores, [-1, kmer_size*encode_size]) ## (B*M,K*E)
            scores = tf.matmul(scores, self.weights1) + self.biases1 ## (K*E,N)
            scores = tf.nn.relu(scores) ## kmers with non-positive scores will be ignored
            ## the shape is (B*M,N)

        with tf.name_scope('gene_layer'):
            scores = tf.reshape(scores, [-1, self.num_motifs]) ## (B*M,N)
            counts = tf.reshape(tf.boolean_mask(counts, valid), [-1, gene_num]) ## (B*M,C)
            scores_expand = tf.expand_dims(scores, axis=2) ## (B*M,N,1)
            counts_expand = tf.expand_dims(counts, axis=1) ## (B*M,1,C)
            scores = tf.matmul(scores_expand, tf.sign(counts_expand)) ## ignore null kmers
            ## the shape is (B*M,N,C)

        with tf.name_scope('max_pooling'):
            bidx = tf.range(batch_size, dtype=tf.int32)
            midx = tf.ones([1, max_kmer], dtype=tf.int32)
            idx = tf.matmul(tf.reshape(bidx, [batch_size, 1]), midx)
            idx = tf.boolean_mask(tf.reshape(idx, [-1]), valid)
            scores = tf.segment_max(scores, idx)
            ## the shape is (B,N,C)
#            scores = tf.nn.l2_normalize(scores, axis=0)

        with tf.name_scope('dropout_layer'):
            scores = tf.nn.dropout(scores, keep_prob)

        with tf.name_scope('motif_layer'):
            scores = scores * self.weights2 ## broadcasting to the first dimention
            scores = tf.reduce_sum(scores, axis=2) + self.biases2 ## (B,N)
            scores = tf.nn.relu(scores) ## select kmers with specific Ig isotypes
            ## the shape is (B,N)

        with tf.name_scope('dropout_layer'):
            scores = tf.nn.dropout(scores, keep_prob)

        return scores

class DeepBCR(GeneSwitchModelFast):
    """ Build up a deep learning model for BCR repertoires

        DeepBCR is an extension to the GeneSwitchModel by adding an optional output layer. 
        And, the DeepBCR can run in three modes:
            1. Classification (such as tumor types or disease stages): num_labels > 1
            2. Linear Regression (such as overall survival): num_labels == 1
            3. Cox-PH Regression (such as hazard ratio): num_labels == -1
            4. Multiple Linear Regression (such as gene expressions): num_labels < -1

        By default, negative values are survival data with censorship (neg_censor=True)
    """

    def __init__(self, num_motifs=30, encode_init=None, num_labels=2, neg_censor=True, model_name='DeepBCR', save_path=''):
        self.model_name = model_name
        self.num_motifs = num_motifs
        self.neg_censor = neg_censor
        self.save_path = save_path
        self.num_labels = num_labels

        if num_labels == 1:
            self.run_mode = 'Linear Regression'
        elif num_labels == -1:
            self.run_mode = 'Cox-PH Regression'
        elif num_labels > 1:
            self.run_mode = 'Classification'
        elif num_labels < -1:
            self.run_mode = 'Multiple Linear Regression'
        else:
            raise ValueError('num_labels must > 1, = 1, = -1, or < -1')
        print(self.model_name, 'runs in', self.run_mode, 'mode.')

        if encode_init is None:
            self.encode_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
            self.encode_shape = (len(AA_LIST), len(AA_LIST))
        elif type(encode_init) is int:
            self.encode_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)
            self.encode_shape = (len(AA_LIST), encode_init)
        else:
            self.encode_init = tf.constant_initializer(encode_init)
            self.encode_shape = encode_init.shape
        print('Encoding matrix has shape', self.encode_shape, 'num_motifs', num_motifs, 'num_labels', num_labels)

    def set_run_mode(self, run_mode):
        options = ['Classification', 'Linear Regression', 'Cox-PH Regression', 'Multiple Linear Regression']
        if run_mode not in options:
            raise ValueError('run_mode '+run_mode+' is not in '+str(options))
        self.run_mode = run_mode
        print(self.model_name, 'runs in', self.run_mode, 'mode.')

    def load_hidden_layer_parameters(self, model_path, xs, cs, ys):
        """ Only load paramters for the hidden layers """
        ys = self.encode_labels(ys) ## encode new labels
        tf.reset_default_graph() ## clean up
        self.pre_train(xs, cs, ys) ## set up the new graph

        saver = tf.train.Saver(self.save_list)

        load_list = [self.weights0, self.weights1, self.biases1, self.weights2, self.biases2]
        loader = tf.train.Saver(load_list)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            loader.restore(session, save_path=model_path)
            print('Load parameters from', model_path)
            self.last_saved_model = saver.save(session, 
                    save_path=os.path.join(self.save_path, self.model_name+'-par/model'), 
                    global_step=0, write_meta_graph=False)
            print('Save parameters to', self.last_saved_model)

    def pre_train(self, xs, cs, ys):
        batch_size, max_kmer, kmer_size = xs.shape
        batch_size, max_kmer, gene_num = cs.shape
        amino_acid, encode_size = self.encode_shape
        with tf.variable_scope('parameters'):
            self.weights0 = tf.get_variable('W0', [amino_acid, encode_size], initializer=self.encode_init)

            self.weights1 = tf.get_variable('W1', [kmer_size*encode_size, self.num_motifs],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases1 = tf.get_variable('b1', [self.num_motifs], initializer=tf.constant_initializer(0.1))

            self.weights2 = tf.get_variable('W2', [self.num_motifs, gene_num],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
            self.biases2 = tf.get_variable('b2', [self.num_motifs], initializer=tf.constant_initializer(0.1))

            if self.run_mode == 'Classification':
                self.weights3 = tf.get_variable('W3', [self.num_motifs, self.num_labels],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
                self.biases3 = tf.get_variable('b3', [self.num_labels], initializer=tf.constant_initializer(0.1))
            elif self.run_mode == 'Linear Regression':
                self.weights3 = tf.get_variable('W3', [self.num_motifs, 1],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
                self.biases3 = tf.get_variable('b3', [1], initializer=tf.constant_initializer(0.1))
            elif self.run_mode == 'Multiple Linear Regression':
                self.weights3 = tf.get_variable('W3', [self.num_motifs, abs(self.num_labels)],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
                self.biases3 = tf.get_variable('b3', [abs(self.num_labels)], initializer=tf.constant_initializer(0.1))
            elif self.run_mode == 'Cox-PH Regression':
                self.weights3 = tf.get_variable('W3', [self.num_motifs, 1],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
                self.biases3 = tf.get_variable('b3', [1], initializer=tf.constant_initializer(0.1))
                # Sort Training Data for Partial Likelihood Calculation
                print('Sort data points by survival time.')
                if self.neg_censor:
                    time = np.abs(ys)
                else:
                    time = ys
                self.ys_sort_idx = np.argsort(time)[::-1]
                xs[:] = xs[self.ys_sort_idx]
                cs[:] = cs[self.ys_sort_idx]
                ys[:] = ys[self.ys_sort_idx]

        self.save_list = [self.weights0, self.weights1, self.biases1, 
                                         self.weights2, self.biases2, 
                                         self.weights3, self.biases3
                         ]
        if hasattr(self, 'only_update_output_layer') and self.only_update_output_layer:
            self.update_list = [self.weights3, self.biases3]

    def post_train(self, xs, cs, ys):
        if hasattr(self, 'update_list'):
            del self.update_list ## clean up since they are graph specific

    def logits(self, features, counts, keep_prob):
        scores = self.hidden_layers(features, counts, keep_prob)

        with tf.name_scope('output_layer'):
            if self.run_mode == 'Classification':
                scores = tf.squeeze(tf.matmul(scores, self.weights3) + self.biases3) ## (N,n)
                scores = tf.reshape(scores, [int(features.get_shape()[0]), -1]) ## fix for one sample case
                ## the shape is (B,n)
            elif self.run_mode == 'Linear Regression':
                scores = tf.matmul(scores, self.weights3) + self.biases3 ## (N,1)
                ## the shape is (B,1)
            elif self.run_mode == 'Multiple Linear Regression':
                scores = tf.squeeze(tf.matmul(scores, self.weights3) + self.biases3) ## (N,n)
                scores = tf.reshape(scores, [int(features.get_shape()[0]), -1]) ## fix for one sample case
                ## the shape is (B,n)
            elif self.run_mode == 'Cox-PH Regression':
                scores = tf.matmul(scores, self.weights3) + self.biases3 ## (N,1)
                ## the shape is (B,1)
        return scores

    def encode_labels(self, ys):
        if self.run_mode == 'Classification':
            ## Expand the integer labels in the multiple label prediction format
            return super(DeepBCR, self).encode_labels(ys)
        elif self.run_mode.endswith('Regression'):
            ## Nothing to do
            return np.array(ys, dtype=float)

    def __correlation(self, x, y):
        var = tf.reduce_mean(x*y) - (tf.reduce_mean(x) * tf.reduce_mean(y))
        ref = tf.sqrt(
            (tf.reduce_mean(tf.square(x)) - tf.square(tf.reduce_mean(x))) *
            (tf.reduce_mean(tf.square(y)) - tf.square(tf.reduce_mean(y))))
        return tf.truediv(var, ref)

    def __negative_log_partial_likelihood(self, risk, censor):
        """ Return the negative log-partial likelihood of the prediction
            
            Modified from DeepSurv model's implementation:
            https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py

            *Note: Need to sort data points by time, which was done in pre_train()

        Parameters:
            risk: predicted risk factors
            censor: 0 means censorship and 1 means event

        Returns:
            neg_likelihood: -log of partial Cox-PH likelihood
        """
        hazard_ratio = tf.exp(risk)
        log_risk = tf.log(tf.cumsum(hazard_ratio))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * censor
        num_observed_events = tf.reduce_sum(censor)
        neg_likelihood = - tf.reduce_sum(censored_likelihood) / num_observed_events
        return neg_likelihood

    def __get_concordance_index(self, risk, time, censor, **kwargs):
        """ Calculate the C-index for evaluating survival models

        Parameters:
            risk: predicted risk factors
            time: survival time
            censor: 0 means censorship and 1 means event

        Returns:
            concordance_index: calcualted using lifelines.utils.concordance_index
        """
        from lifelines.utils import concordance_index
        partial_hazard = tf.exp(-risk)
        return concordance_index(time, partial_hazard, censor) ## TODO: translate to TF operations

    def objective(self, logits, labels):
        """ get the objective function """
        if self.run_mode == 'Classification':
            return super(DeepBCR, self).objective(logits, labels)
        elif self.run_mode == 'Multiple Linear Regression':
            with tf.name_scope('objective'):
                err = tf.reduce_mean(tf.square(labels-logits), axis=0) ## Squared error
#                re1 = tf.reduce_mean(tf.square(self.weights3), axis=0) ## Tikhonov regularization
#                re2 = tf.reduce_mean(tf.square(self.biases3))
                re1 = tf.reduce_mean(tf.abs(self.weights3), axis=0) ## L1 regularization
                re2 = 0
                return err + re1 + re2
        elif self.run_mode == 'Linear Regression':
            with tf.name_scope('objective'):
                err = tf.reduce_mean(tf.square(labels-logits)) ## Squared error
#                re1 = tf.reduce_mean(tf.square(self.weights3)) ## Tikhonov regularization
#                re2 = tf.reduce_mean(tf.square(self.biases3))
                re1 = tf.reduce_mean(tf.abs(self.weights3)) ## L1 regularization
                re2 = 0
                return err + re1 + re2
        elif self.run_mode == 'Cox-PH Regression':
            if self.neg_censor:
                censor = tf.sign(labels)
            else:
                censor = tf.ones(labels.get_shape())
            pl = self.__negative_log_partial_likelihood(logits, censor)
            return pl

    def accuracy(self, logits, labels):
        """ Get the prediction accuracy or similar measures which differ from the objective """

        if self.run_mode == 'Classification':
            return super(DeepBCR, self).accuracy(logits, labels)
        elif self.run_mode == 'Multiple Linear Regression':
            with tf.name_scope('accuracy'):
                r = self.__correlation(logits, labels) ## TODO: r for each label
                tf.summary.scalar('summary', r)
                return r
        elif self.run_mode == 'Linear Regression':
            with tf.name_scope('accuracy'):
                r = self.__correlation(logits, labels)
                tf.summary.scalar('summary', r)
                return r
        elif self.run_mode == 'Cox-PH Regression':
            with tf.name_scope('accuracy'):
                if self.neg_censor:
                    time = tf.abs(labels)
                    censor = tf.sign(labels)
                else:
                    time = labels
                    censor = tf.ones(labels.get_shape())
                ## TODO: use C-index
                c = self.__get_concordance_index(logits, time, censor)
                tf.summary.scalar('summary', c)
                return c

    def performance(self, out):
        if self.run_mode == 'Classification':
            return super(DeepBCR, self).performance(out)
        elif self.run_mode.endswith('Regression'): ## TODO: skip Cox-PH
            print(self.model_name, 'Objective:', '%4.3f'%np.mean(out[1]), end=' ')
            print('Correlation:', '%4.1f'%(100.0*out[2]))
            return out[1], out[2]
        elif self.run_mode == 'Cox-PH Regression':
            return out[1], out[2]

def get_syn_data(num_samples=30, num_kmers=30, kmer_size=3, num_pos_kmers=3, aa_list=AA_LIST, positive=None):
    """ Function for generating the synthetic data for deep_bcr models.

        Modified from dataplumbing_synthetic_data.py in https://github.com/jostmey/MaxSnippetModel

        Args:
            num_samples: number of samples in the batch
            num_kmers: maximum number of kmers in each sample
            kmer_size: the size of k-mer, i.e.: k
            num_pos_kmers: number of recurrent kmers in positive samples
            aa_list: the list of amino acids used for creating k-mers
            positive: the list of positive kmers. If None, will generate based on num_pos_kmers

        Returns:
            xs: encoded repertoires, [num_samples, num_kmers, kmer_size]
                where each amino acid is represented as itegers
            cs: indication of the missing data [num_samples, num_kmers]
            ys: labels for positive or negative samples [num_samples]
            positive: the list of positive kmers used in generating the data
    """
    aa_size = len(aa_list)

    xs = np.zeros((num_samples, num_kmers, kmer_size), dtype=int) # Features
    cs = np.zeros((num_samples, num_kmers), dtype=int) # Kmer count
    ys = np.zeros((num_samples), dtype=int) # Labels

    for i in range(num_samples):
        N = np.random.randint(num_kmers//2)+num_kmers//2-1
        for j in range(N):
            cs[i,j] = 1.0
            for k in range(kmer_size):
                xs[i,j,k] = np.random.randint(aa_size)

    if positive is None:
        positive = []
        for i in range(num_pos_kmers):
            kmer = ''
            for k in range(kmer_size):
                kmer += aa_list[np.random.randint(aa_size)]
            positive.append(kmer)
        print('Positive kmers:', positive)

    for i in range(round(num_samples/2)):
        ys[i] = 1.0
        kmer = positive[np.random.randint(len(positive))]
        j = np.random.randint(0,num_kmers//2,1)[0]
        cs[i,j] = 1.0
        for k in range(len(kmer)):
            xs[i,j,k] = aa_list.find(kmer[k])

    return xs, cs, ys, positive

def index_to_binary(xs, aa_list=AA_LIST, flat=True):
    """ Transform a index matrix into a binary matrix

        For example, if aa_list='ABCD' and flat=False:
            [0 1]     [[1 0 0 0], [0 1 0 0]]
            [3 0] --> [[0 0 0 1], [1 0 0 0]]
            [2 1]     [[0 0 1 0], [0 1 0 0]]
    """
    codes = np.eye(len(AA_LIST))
    dims = list(xs.shape)
    x = xs.reshape(-1)
    y = codes[x]
    if flat: # expand the last dimention
        dims[-1] = -1
    else: # push to the extra dimention
        dims.append(len(AA_LIST))
    return y.reshape(tuple(dims))

if __name__ == '__main__':
    save_path = '../work/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    from datetime import datetime
    START_TIME = datetime.now()
    res = []

    n = 200
    RE = False # whether to recovere saved results
    SL = False # whether to save log files

    for m in [10, 50, 100]:
        for i in [1,2,4,8,16]:
            model_path = os.path.join(save_path, 'num%s_max%s_pos%s/'%(n,m,i))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            print(m, '--------------', i)
            xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, num_pos_kmers=i)

            print('Train data', xs.shape, cs.shape, ys.shape)
            XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, num_pos_kmers=i, positive=ref)
            print('Test data', XS.shape, CS.shape, YS.shape)

            m1 = MaxSnippetModel(save_path=model_path)
            t1 = m1.train(index_to_binary(xs), cs, ys, save_log=SL, recover=RE)
            r1 = m1.test(index_to_binary(XS), CS, YS)
            m1.predict(index_to_binary(XS), CS)
            res.append((n,m,i,m1.model_name, r1[1][m1.bestfit_index], r1[2][m1.bestfit_index]))

            m2 = TwoLayerModel(save_path=model_path)
            t2 = m2.train(index_to_binary(xs), cs, ys, save_log=SL, recover=RE)
            r2 = m2.test(index_to_binary(XS), CS, YS)
            m2.predict(index_to_binary(XS), CS)
            res.append((n,m,i,m2.model_name,r2[1],r2[2][0]))

            m3 = MultipleLabelModel(save_path=model_path)
            t3 = m3.train(index_to_binary(xs), cs, ys, save_log=SL, recover=RE)
            r3 = m3.test(index_to_binary(XS), CS, YS)
            m3.predict(index_to_binary(XS), CS)
            res.append((n,m,i,m3.model_name,r3[1],r3[2]))

            m4 = EncodingLayerModel(save_path=model_path)
            t4 = m4.train(xs, cs, ys, save_log=SL, recover=RE)
            r4 = m4.test(XS, CS, YS)
            m4.predict(XS, CS)
            res.append((n,m,i,m4.model_name,r4[1],r4[2]))

            m5 = GeneSwitchModelFast(save_path=model_path)
            t5 = m5.train(xs, np.tile(cs.reshape(n,m,1),(1,1,2)), ys, save_log=SL, recover=RE)
            r5 = m5.test(XS, np.tile(CS.reshape(n,m,1),(1,1,2)), YS)
            m5.predict(XS, np.tile(CS.reshape(n,m,1),(1,1,2)))
            res.append((n,m,i,m5.model_name,r5[1],r5[2]))

            m6 = DeepBCR(num_labels=2, save_path=model_path)
            t6 = m6.train(xs, np.tile(cs.reshape(n,m,1),(1,1,2)), ys, save_log=SL, recover=False)
            r6 = m6.test(XS, np.tile(CS.reshape(n,m,1),(1,1,2)), YS)
            m6.predict(XS, np.tile(CS.reshape(n,m,1),(1,1,2)))
            res.append((n,m,i,m6.model_name,r6[1],r6[2]))

            break  ## no need for a full test
        break

    res = pd.DataFrame(res, columns=['#Samples','#Snips','#PosCases','Model','Test_obj','Test_acc'])
    res.to_csv(os.path.join(save_path, 'result_compare.csv'), index=False)
    print('Models are saved in', save_path)

    FINISH_TIME = datetime.now()
    print('Start  at', START_TIME)
    print('Finish at', FINISH_TIME)
    print("Time Cost", FINISH_TIME-START_TIME)

