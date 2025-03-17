import numpy as np


class Optimizer:
    def __init__(self, params, optimizer_name, learning_rate, momentum=0.5,
                 beta=0.5, beta1=0.5, beta2=0.5, epsilon=1e-6, weight_decay=0.0):
        
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # timestep for Adam/Nadam
        self.t = 0  
        
        # Initialize optimizer states
        self.v = {}  # For momentum or Adam's first moment
        self.s = {}  # For RMSProp or Adam's second moment
        
        for key, val in params.items():
            self.v[key] = np.zeros_like(val)
            self.s[key] = np.zeros_like(val)
        self.param_keys = list(params.keys())
    
    def update(self, params, grads):
        self.t += 1
        
        for key in self.param_keys:
            grad = grads['d' + key]
            
            # Applying weight decay to weights only
            if key.startswith('W'):
                grad += self.weight_decay * params[key]
            
            if self.optimizer_name == 'sgd':
                params[key] -= self.learning_rate * grad
                
            elif self.optimizer_name == 'momentum':
                self.v[key] = self.momentum * self.v[key] - self.learning_rate * grad
                params[key] += self.v[key]
                
            
            elif self.optimizer_name in ['nag', 'nesterov']:
                v_prev = self.v[key].copy()
                self.v[key] = self.momentum * self.v[key] - self.learning_rate * grad
                params[key] += -self.momentum * v_prev + (1 + self.momentum) * self.v[key]
                
            
            elif self.optimizer_name == 'rmsprop':
                self.s[key] = self.beta * self.s[key] + (1 - self.beta) * (grad ** 2)
                params[key] -= self.learning_rate * grad / (np.sqrt(self.s[key]) + self.epsilon)
                
            
            elif self.optimizer_name == 'adam':
                self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grad
                self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grad ** 2)
                v_corr = self.v[key] / (1 - self.beta1 ** self.t)
                s_corr = self.s[key] / (1 - self.beta2 ** self.t)
                params[key] -= self.learning_rate * v_corr / (np.sqrt(s_corr) + self.epsilon)
                
            
            elif self.optimizer_name == 'nadam':
                self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grad
                self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grad ** 2)
                v_corr = self.v[key] / (1 - self.beta1 ** self.t)
                s_corr = self.s[key] / (1 - self.beta2 ** self.t)
                params[key] -= self.learning_rate * (self.beta1 * v_corr + (1 - self.beta1) * grad) / (np.sqrt(s_corr) + self.epsilon)
                
            else:
                raise ValueError("Unknown optimizer: " + self.optimizer_name)
