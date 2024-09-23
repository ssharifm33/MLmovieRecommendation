'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np
from autograd.numpy.numpy_boxes import ArrayBox
import numpy as np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets


# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        
        random_state = self.random_state # inherited RandomState object
        n_factors = self.n_factors

        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users),
            c_per_item=ag_np.ones(n_items),
            U=ag_np.array(0.01 * random_state.randn(n_users, n_factors)),
            V=ag_np.array(0.01 * random_state.randn(n_items, n_factors)),
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        yhat = ag_np.zeros(user_id_N.size)

        if mu is None:
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict['b_per_user']
        if c_per_item is None:
            c_per_item = self.param_dict['c_per_item']
        if U is None:
            U = self.param_dict['U']
        if V is None:
            V = self.param_dict['V']

        # indexing with user_ID and item_ID
        yhat = (mu + b_per_user[user_id_N] + c_per_item[item_id_N] +  ag_np.sum(U[user_id_N]*V[item_id_N], axis=1))
        return yhat


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''

        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        user_cheds = ag_np.sum(param_dict['U'][data_tuple[0]]**2)
        item_cheds = ag_np.sum(param_dict['V'][data_tuple[1]]**2)

        first_chunk = self.alpha*(user_cheds + item_cheds)
        loss_total = first_chunk + ag_np.sum((y_N - yhat_N)**2)
        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=2, alpha=0.5)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)