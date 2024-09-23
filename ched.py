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
        
        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions

        n_factors = self.n_factors

        self.param_dict = dict(
        mu = 0.01 * random_state.randn(1),  # Initialize to small random values
        b_per_user = 0.01 * random_state.randn(n_users),  # Bias per user
        c_per_item = 0.01 * random_state.randn(n_items),  # Bias per item
        U = 0.01 * random_state.randn(n_users, n_factors),  # User factors
        V = 0.01 * random_state.randn(n_items, n_factors)  # Item factors
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
        N = user_id_N.size


        # U -> USERS   (# users x n_factors)
        # V -> ITEMS   (# items x n_factors)

        # Prediction formula: r_ui = mu + b_u + b_i + q_i^T*p_u

            # - mu = None
            # - b_u = b_per_user
        
        # FROM SLIDES: y_hat = mu + b_i + c_j + SUM{K=1 to K} u_ik * v_jk

            # K = n_factors
        
        yhat = ag_np.zeros(N)

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

        # for i in range(N):
        #     sum = 0
        #     curr_user_id = user_id_N[i]
        #     curr_item_id = item_id_N[i]
        #     for j in  range(self.n_factors):
        #         sum += U[curr_user_id][j] * V[curr_item_id][j]    

        #     total = mu + b_per_user[curr_user_id] + c_per_item[curr_item_id] + sum
            
        #     if isinstance(total, ArrayBox):
        #         total = float(total._value[0])
        #     elif isinstance(total, np.ndarray):
        #         total = total[0]
        #     else:
        #         return "Unknown type"
            
        #     # print(total)
        #     yhat[i] = total
        # print("ched")
        # print(len(yhat))
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
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength

        # U: N * K
        # V: M * K

        # Equation: a( sum{sum{v_jk^2}} + sum{})

        # num_users = len(self.param_dict['U'])
        # num_items = len(self.param_dict['V'])
        # num_factors = self.n_factors

        # k = num_factors
        # j = num_items
        # i = num_users



        y_N = data_tuple[2]  # Actual ratings
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)  # Predicted ratings

        # Regularization terms
        reg_term = self.alpha * (ag_np.sum(param_dict['U']**2) + ag_np.sum(param_dict['V']**2))

        # Mean squared error - using ag_np instead of np
        mse = ag_np.mean((y_N - yhat_N)**2)

        # Total loss
        return mse + reg_term




        # y_N = data_tuple[2]
        # yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        # user_cheds = ag_np.sum(self.param_dict['U'][data_tuple[0]]**2)
        # item_cheds = ag_np.sum(self.param_dict['V'][data_tuple[0]]**2)

        # first_chunk = self.alpha*(user_cheds + item_cheds)
        # loss_total = first_chunk + ag_np.mean((y_N - yhat_N)**2)
        # return loss_total    


# if __name__ == '__main__':

#     # Load the dataset
#     train_tuple, valid_tuple, test_tuple, n_users, n_items = \
#         load_train_valid_test_datasets()
#     # Create the model and initialize its parameters
#     # to have right scale as the dataset (right num users and items)
#     model = CollabFilterOneVectorPerItem(
#         n_epochs=10, batch_size=10000, step_size=0.1,
#         n_factors=2, alpha=0.0)
#     model.init_parameter_dict(n_users, n_items, train_tuple)   # Train Tuple: 3 x 70,000

#     # Fit the model with SGD
#     model.fit(train_tuple, valid_tuple)

if __name__ == '__main__':
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

    # Create the model
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=1000, step_size=20,  # Adjusted for potentially better convergence
        n_factors=10, alpha=0.5  # More factors and non-zero regularization
    )

    # Initialize parameters
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD and monitor progress
    model.fit(train_tuple, valid_tuple)