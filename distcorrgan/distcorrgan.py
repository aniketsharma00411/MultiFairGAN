import torch
import torch.nn.functional as f
from torch import nn
import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


def get_ohe_data(df, S, Y, S_under, Y_desire):
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(
        df.select_dtypes(['float', 'integer']).columns)
    ##############################################################
    scaler = QuantileTransformer(
        n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object')
    df_cat_names = list(df.select_dtypes('object').columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    S_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
    Y_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])

    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0] == S_under:
        underpriv_index = 0
        priv_index = 1
    else:
        underpriv_index = 1
        priv_index = 0
    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
        desire_index = 0
        undesire_index = 1
    else:
        desire_index = 1
        undesire_index = 0

    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index


def get_original_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(
        ['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(
        ['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(
        ['float', 'integer']).columns)
    df_cat = pd.DataFrame(
        df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)


def prepare_data(df, batch_size, S, Y, S_under, Y_desire):
    ohe, scaler, discrete_columns, continuous_columns, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data(
        df, S, Y, S_under, Y_desire)
    input_dim = df_transformed.shape[1]
    X_train, X_test = train_test_split(
        df_transformed, test_size=0.1, shuffle=True)
    data_train = X_train.copy()
    data_test = X_test.copy()

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    data = torch.from_numpy(data_train).float()

    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    return ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index


class Generator(nn.Module):
    def __init__(self, input_dim, continuous_columns, discrete_columns):
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._discrete_columns = discrete_columns
        self._num_continuous_columns = len(continuous_columns)

        self.lin1 = nn.Linear(self._input_dim, self._input_dim)
        self.lin_numerical = nn.Linear(
            self._input_dim, self._num_continuous_columns)

        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._input_dim, value)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        # x = f.leaky_relu(self.lin1(x))
        # x_numerical = f.leaky_relu(self.lin_numerical(x))
        x_numerical = f.relu(self.lin_numerical(x))
        x_cat = []
        for key in self.lin_cat:
            x_cat.append(f.gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self._input_dim = input_dim
        # self.dense1 = nn.Linear(109, 256)
        self.dense1 = nn.Linear(self._input_dim, self._input_dim)
        self.dense2 = nn.Linear(self._input_dim, self._input_dim)
        # self.dense3 = nn.Linear(256, 1)
        # self.drop = nn.Dropout(p=0.2)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = f.leaky_relu(self.dense1(x))
        # x = self.drop(x)
        # x = f.leaky_relu(self.dense2(x))
        x = f.leaky_relu(self.dense2(x))
        # x = self.drop(x)
        return x


class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        I = x[:, self._Y_start_index:self._Y_start_index + 2]

        disp = -1.0 * lamda * \
            self.compute_distance_correlation(
                G, I) - 1.0 * torch.mean(crit_fake_pred)

        return disp

    # Source: https://gist.github.com/amirarsalan90/b4c975f2b15589c4b0c57377e86b79a7
    # The calculation is based on the following paper:
    # Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing dependence by correlation of distances. The #annals of statistics, 35(6), 2769-2794.
    def compute_distance_covariance(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        dist_A = torch.cdist(A, A, p=1)
        dist_B = torch.cdist(B, B, p=1)

        row_mean_A = torch.mean(dist_A, axis=1)
        row_mean_B = torch.mean(dist_B, axis=1)

        dist_A_rowcentered = dist_A - \
            row_mean_A.expand((A.shape[0], A.shape[0])).T
        dist_B_rowcentered = dist_B - \
            row_mean_B.expand((A.shape[0], A.shape[0])).T

        dist_A_rowcentered_colcentered = dist_A_rowcentered - \
            row_mean_A.expand((A.shape[0], A.shape[0]))
        dist_B_rowcentered_colcentered = dist_B_rowcentered - \
            row_mean_B.expand((A.shape[0], A.shape[0]))

        dist_A_rowcentered_colcentered = dist_A_rowcentered_colcentered + \
            torch.mean(dist_A)
        dist_B_rowcentered_colcentered = dist_B_rowcentered_colcentered + \
            torch.mean(dist_B)
        cov2 = torch.dot(dist_A_rowcentered_colcentered.flatten(),
                         dist_B_rowcentered_colcentered.flatten())

        avgcov2 = cov2 / (A.shape[0]**2)
        return avgcov2

    def compute_distance_correlation(self, A, B):
        distcovAB = self.compute_distance_covariance(A, B)
        covA = torch.sqrt(self.compute_distance_covariance(A, A))
        covB = torch.sqrt(self.compute_distance_covariance(B, B))

        distcor = distcovAB / (covA * covB)

        return distcor


device = torch.device("cuda:0" if (
    torch.cuda.is_available() and 1 > 0) else "cpu")


def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - \
        torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss


def train(df, S, Y, S_under, Y_desire, epochs=500, batch_size=64, lamda=0.5):
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = prepare_data(
        df, batch_size, S, Y, S_under, Y_desire)

    generator = Generator(input_dim, continuous_columns,
                          discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    second_critic = FairLossFunc(
        S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(device)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(
        generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(
        critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss = nn.BCELoss()
    critic_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        print("epoch {}".format(i + 1))
        ############################
        print("training for fairness")
        for data in train_dl:
            data[0] = data[0].to(device)
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(
                    size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(batch_size, input_dim,
                                     device=device, requires_grad=True)
                gradient = get_gradient(
                    critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                crit_loss = get_crit_loss(
                    crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]

            ###############################
            # training the generator for fairness
            gen_optimizer_fair.zero_grad()
            fake_noise_2 = torch.randn(
                size=(batch_size, input_dim), device=device).float()
            fake_2 = generator(fake_noise_2)

            crit_fake_pred = critic(fake_2)

            gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
            gen_fair_loss.backward()
            gen_optimizer_fair.step()
            cur_step += 1

    return generator, critic, ohe, scaler, data_train, data_test, input_dim


def train_plot(df, S, Y, S_under, Y_desire, epochs, lamda, batchsize=256):
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(
        df, S, Y, S_under, Y_desire, epochs, batchsize, lamda)
    return generator, critic, ohe, scaler, data_train, data_test, input_dim
