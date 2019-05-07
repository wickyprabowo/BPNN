import numpy as np
import math


class backpropagation(object):
    def __init__(self, input, hidden, output, alpha, max_epoch):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.alpha = alpha
        self.max_epoch = max_epoch

    def training(self, train_data, train_class):
        beta = self.hitung_beta(self.input, self.output)

        self.weight_hidden = np.random.uniform(-0.5, 0.5, size=(self.input, self.hidden))

        self.v_lama = self.widrow_v_lama(self.input, self.hidden, self.weight_hidden)

        self.v_baru = self.widrow_v_baru(self.input, self.hidden, beta, self.weight_hidden, self.v_lama)

        self.bias_hidden = np.random.uniform(-beta, beta, size=(self.hidden))

        self.weight_output = np.random.uniform(-0.5, 0.5, size=(self.hidden, self.output))
        self.bias_output = np.random.uniform(-0.5, 0.5, size=(self.output))

        print(self.v_baru)
        print(self.bias_hidden)
        print(self.weight_output)
        print(self.bias_output)

        n = 0
        while n < self.max_epoch:
            for data, target in zip(train_data, train_class):
                # print(data)
                # print(target)

                z_in = np.dot(data, self.v_baru) + self.bias_hidden
                z = np.array([])
                for i in z_in:
                    z = np.append(z, self.aktivasi(i))
                # print("Z in : ", z_in)
                # print("Z : ", z)

                y_in = np.dot(z, self.weight_output) + self.bias_output
                y = np.array([])
                for j in y_in:
                    y = np.append(y, self.aktivasi(j))
                # print("Y in : ", y_in)
                # print("Y : ", y)

                s_output = np.array([])
                for target, y in zip(target, y):
                    s_output = np.append(s_output, self.faktor_error_output(target, y))
                # print("S Ouput : ", s_output)

                delta_bobot_w = self.delta_bobot_w(self.hidden, self.output, self.alpha, s_output, z)
                # print("Delta bobot W : ", delta_bobot_w)

                delta_bobot_bias_w = self.delta_bobot_bias_w(self.output, self.alpha, s_output)
                # print("Delta bobot Bias di W : ", delta_bobot_bias_w)

                s_in_hidden = np.dot(self.weight_output, s_output)
                # print("S in Hidden", s_in_hidden)

                s_hidden = np.array([])
                for s_in_hidden, z in zip(s_in_hidden, z):
                    s_hidden = np.append(s_hidden, self.faktor_error_hidden(s_in_hidden, z))
                # print("S Hidden : ", s_hidden)

                delta_bobot_v = self.delta_bobot_v(self.input, self.hidden, self.alpha, s_hidden, data)
                # print("Delta bobot V : ", delta_bobot_v)

                delta_bobot_bias_v = self.delta_bobot_bias_v(self.hidden, self.alpha, s_hidden)
                # print("Delta bobot Bias di V : ", delta_bobot_bias_v)

                update_bobot_v = self.update_bobot_v(delta_bobot_v)
                # print("V : ", update_bobot_v)

                update_bobot_bias_v = self.update_bobot_bias_v(delta_bobot_bias_v)
                # print("Bias V : ", update_bobot_bias_v)

                update_bobot_w = self.update_bobot_w(delta_bobot_w)
                # print("W : ", update_bobot_w)

                update_bobot_bias_w = self.update_bobot_bias_w(delta_bobot_bias_w)
                # print("Bias W : ", update_bobot_bias_w)

            n += 1
        print(n)
        print(self.v_baru)
        print(self.bias_hidden)
        print(self.weight_output)
        print(self.bias_output)

    def testing(self, testing_data, testing_class):

        correct = 0
        accuracy = 0
        total_data = 0

        for data, target in zip(testing_data, testing_class):
            z_in = np.dot(data, self.v_baru) + self.bias_hidden
            z = np.array([])
            for i in z_in:
                z = np.append(z, self.aktivasi(i))
            print("Z in : ", z_in)
            print("Z : ", z)

            y_in = np.dot(z, self.weight_output) + self.bias_output
            y = np.array([])
            for j in y_in:
                y = np.append(y, self.aktivasi(j))
            y_new = np.array([])
            for k in y:
                if k >= 0.5:
                    y_new = np.append(y_new, 1)
                else:
                    y_new = np.append(y_new, 0)
            total_data += 1
            if self.check(target, y_new):
                correct += 1
                accuracy = (correct / total_data) * 100

            print("Y in : ", y_in)
            print("Y : ", y)
            print("Y : ", y_new)
            print("Accuracy : ", accuracy)
            print(correct, total_data)

    def hitung_beta(self, input, output):
        beta = 0.7 * math.pow(output, (1 / input))
        return beta

    def widrow_v_lama(self, input, hidden, weight_hidden):
        v_lama = []
        for i in range(hidden):
            v_old = 0
            for j in range(input):
                v_old += math.pow(weight_hidden[j][i], 2)
            v_old_sqrt = math.sqrt(v_old)
            v_lama.append(v_old_sqrt)
        return v_lama

    def widrow_v_baru(self, input, hidden, beta, weight_hidden, v_lama):
        v_baru = []
        for i in range(input):
            v_baru.append([])
            for j in range(hidden):
                v_new = (beta * weight_hidden[i][j]) / v_lama[j]
                v_baru[i].append(v_new)
        return v_baru

    def aktivasi(self, nilai):
        aktiv = 1 / (1 + np.exp(-(nilai)))
        return aktiv

    def faktor_error_output(self, target, y):
        s_ouput = (target - y) * (y * (1 - y))
        return s_ouput

    def faktor_error_hidden(self, s_in_hidden, z):
        s_hidden = s_in_hidden * z * (1 - z)
        return s_hidden

    def delta_bobot_w(self, hidden, output, alpha, faktor_error_output, z):
        delta_w = []
        for i in range(hidden):
            delta_w.append([])
            for j in range(output):
                w = alpha * faktor_error_output[j] * z[i]
                delta_w[i].append(w)
        return delta_w

    def delta_bobot_bias_w(self, output, alpha, faktor_error):
        delta_bias_w = []
        for j in range(output):
            bias_w = alpha * faktor_error[j]
            delta_bias_w.append(bias_w)
        return delta_bias_w

    def delta_bobot_v(self, input, hidden, alpha, faktor_error_hidden, x):
        delta_v = []
        for i in range(input):
            delta_v.append([])
            for j in range(hidden):
                v = alpha * faktor_error_hidden[j] * x[i]
                delta_v[i].append(v)
        return delta_v

    def delta_bobot_bias_v(self, hidden, alpha, faktor_error_hidden):
        delta_bias_v = []
        for j in range(hidden):
            bias_v = alpha * faktor_error_hidden[j]
            delta_bias_v.append(bias_v)
        return delta_bias_v

    def update_bobot_w(self, delta_bobot_w):
        for i in range(self.hidden):
            for j in range(self.output):
                self.weight_output[i][j] = self.weight_output[i][j] + delta_bobot_w[i][j]
        return self.weight_output

    def update_bobot_bias_w(self, delta_bobot_bias_w):
        for i in range(self.output):
            self.bias_output[i] = self.bias_output[i] + delta_bobot_bias_w[i]
        return self.bias_output

    def update_bobot_v(self, delta_bobot_v):
        for i in range(self.input):
            for j in range(self.hidden):
                self.v_baru[i][j] = self.v_baru[i][j] + delta_bobot_v[i][j]
        return self.v_baru

    def update_bobot_bias_v(self, delta_bobot_bias_v):
        for i in range(self.hidden):
            self.bias_hidden[i] = self.bias_hidden[i] + delta_bobot_bias_v[i]
        return self.bias_hidden

    def check(self, target, y_new):
        for i, j in zip(target, y_new):
            if i != j:
                return False
        return True
