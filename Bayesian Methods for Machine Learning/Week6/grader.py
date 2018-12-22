import requests
import json
import numpy as np
from collections import OrderedDict

class Grader(object):
    def __init__(self):
        self.submission_page = 'https://hub.coursera-apps.org/api/onDemandProgrammingScriptSubmissions.v1'
        self.assignment_key = 'ZJzC93UJEeesww5LLQnVZg'
        self.parts = OrderedDict([('P8Xj7', '1.1'), 
                      ('sYdjs', '1.2 (mean)'), 
                      ('Mjy6R', '1.2 (variance)'),
                      ('Wif7t', '1.3'),
                      ('V9yZN', '1.4 (noise)'),
                      ('s4es0', '1.4 (just signal)'),
                      ('ckZSh', '1.5'),
                      ('1Jngf', '2.1'),
                      ('CBiGW', '2.2')])
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

    def submit_GPy_1(self, output):
        self.submit_part('P8Xj7', str(self.ravel_output(output)))

    def submit_GPy_2(self, mean, var):
        self.submit_part('sYdjs', str(self.ravel_output(mean)))
        self.submit_part('Mjy6R', str(self.ravel_output(var)))
        
    def submit_GPy_3(self, output):
        self.submit_part('Wif7t', str(self.ravel_output(output)))

    def submit_GPy_4(self, noise, just_signal):
        self.submit_part('V9yZN', str(self.ravel_output(noise)))
        self.submit_part('s4es0', str(self.ravel_output(just_signal)))
        
    def submit_GPy_5(self, output):
        self.submit_part('ckZSh', str(self.ravel_output(output))) 
        
    def submit_GPyOpt_1(self, output):
        self.submit_part('1Jngf', str(self.ravel_output(output)))
        
    def submit_GPyOpt_2(self, output):
        self.submit_part('CBiGW', str(self.ravel_output(output)))
