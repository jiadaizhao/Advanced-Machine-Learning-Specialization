import requests
import json
import numpy as np
from collections import OrderedDict
from keras.layers import Input
import tensorflow as tf
from keras.datasets import mnist

class Grader(object):
    def __init__(self):
        self.submission_page = 'https://hub.coursera-apps.org/api/onDemandProgrammingScriptSubmissions.v1'
        self.assignment_key = 'Pf_j7noDEeexdQ4iFFMrvA'
        self.parts = OrderedDict([('S66Mi', '1 (vlb)'),
                                  ('dXfpy', '2.1 (samples mean)'),
                                  ('U1gJG', '2.2 (samples var)'),
                                  ('NRPCA', '3 (best val loss)'),
                                  ('JEmpp', '4.1 (hallucinating mean)'),
                                  ('3K3IB', '4.2 (hallucinating var)'),
                                  ('tYD01', '5.1 (conditional hallucinating mean)'),
                                  ('CaofU', '5.2 (conditional hallucinating var)'),])
        self.answers = {key: None for key in self.parts}

    @staticmethod
    def ravel_output(output):
        '''
           If student accedentally submitted np.array with one
           element instead of number, this function will submit
           this number instead
        '''
        if isinstance(output, np.ndarray) and output.size == 1:
            output = output.item(0)
        return output

    def submit(self, email, token):
        submission = {
                    "assignmentKey": self.assignment_key, 
                    "submitterEmail": email, 
                    "secret": token, 
                    "parts": {}
                  }
        for part, output in self.answers.items():
            if output is not None:
                submission["parts"][part] = {"output": output}
            else:
                submission["parts"][part] = dict()
        request = requests.post(self.submission_page, data=json.dumps(submission))
        response = request.json()
        if request.status_code == 201:
            print('Submitted to Coursera platform. See results on assignment page!')
        elif u'details' in response and u'learnerMessage' in response[u'details']:
            print(response[u'details'][u'learnerMessage'])
        else:
            print("Unknown response from Coursera: {}".format(request.status_code))
            print(response)

    def status(self):
        print("You want to submit these numbers:")
        for part_id, part_name in self.parts.items():
            answer = self.answers[part_id]
            if answer is None:
                answer = '-'*10
            print("Task {}: {}".format(part_name, answer))
               
    def submit_part(self, part, output):
        self.answers[part] = output
        print("Current answer for task {} is: {}".format(self.parts[part], output))

    def submit_vlb(self, sess, vlb_binomial):
        test_data = np.load('test_data.npz')
        my_x = Input(batch_shape=(100, 784))
        my_x_decoded = Input(batch_shape=(100, 784))
        my_t_mean = Input(batch_shape=(100, 2))
        my_t_log_var = Input(batch_shape=(100, 2))
        loss = vlb_binomial(my_x, my_x_decoded, my_t_mean, my_t_log_var)
        try:
            output = sess.run(loss, feed_dict={my_x: test_data['x'], my_x_decoded: test_data['x_decoded_mean'],
                              my_t_mean: test_data['t_mean'], my_t_log_var: test_data['t_log_var']})
        except Exception as e:
            print('Sorry, we were not able to run the provided code in `sess`.')
            raise e
        self.submit_part('S66Mi', str(self.ravel_output(output)))

    def submit_samples(self, sess, sampling):
        test_data = np.load('test_data.npz')
        my_t_mean = tf.tile(test_data['t_mean'][:1, :], [10000, 1])
        my_t_log_var = tf.tile(test_data['t_log_var'][:1, :], [10000, 1])
        samples = sampling([my_t_mean, my_t_log_var])
        try:
            samples = sess.run(samples)
        except Exception as e:
            print('Sorry, we were not able to run the provided code in `sess`.')
            raise e
        mean = np.mean(samples, axis=0)[1]
        var = np.var(samples, axis=0)[1]
        self.submit_part('dXfpy', str(self.ravel_output(mean)))
        self.submit_part('U1gJG', str(self.ravel_output(var)))

    def submit_best_val_loss(self, hist):
        self.submit_part('NRPCA', str(self.ravel_output(hist.history['val_loss'][-1])))

    def submit_hallucinating(self, sess, sampled_im_mean):
        try:
            imgs = sess.run(sampled_im_mean)
        except Exception as e:
            print('Sorry, we were not able to run the provided code in `sess`.')
            raise e
        self.submit_part('JEmpp', str(self.ravel_output(np.mean(imgs))))
        var_per_channel = np.var(imgs, axis=0)
        self.submit_part('3K3IB', str(self.ravel_output(np.max(var_per_channel))))

    def submit_conditional_hallucinating(self, sess, conditional_sampled_im_mean):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        
        baseline = np.zeros((10, 784))
        for i in range(10):
            idx = y_train == i
            baseline[i, :] = np.mean(x_train[idx, :], axis=0)
        baseline_repeated = np.repeat(baseline, 5, axis=0)
        
        try:
            imgs = sess.run(conditional_sampled_im_mean)
        except Exception as e:
            print('Sorry, we were not able to run the provided code in `sess`.')
            raise e
            
        diff = np.abs(imgs - baseline_repeated)
        self.submit_part('tYD01', str(self.ravel_output(np.mean(diff))))
        var_per_channel = np.var(diff, axis=0)
        self.submit_part('CaofU', str(self.ravel_output(np.max(var_per_channel))))
