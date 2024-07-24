import numpy as np
import cv2
import scipy.special
import scipy.ndimage
import re

import matplotlib.pyplot as plt
import gzip
import os, sys

# neural network class definition
class NeuralNetwork:

    #initial the neural network
    def __init__(self, input, hidden, output, lr, act_fun = "Signoid"):
        # set number of nodes in each input, hidden, output layer
        self.inodes = input
        self.hnodes = hidden
        self.onodes = output

        # learning rate
        self.lr = lr

        self.act_fun = act_fun

        #self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        #self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        if self.act_fun == "Signoid":
            
            self.activation_function = lambda x: scipy.special.expit(x)
            
            self.inverse_activation_function = lambda x: scipy.special.logit(x)

        elif self.act_fun == "Relu":

            self.activation_function = lambda x: np.where(x > 0, x, 0.01 * x)
            
            self.inverse_activation_function = lambda x: np.where(x > 0, x, 100 * x)
            
            self.activation_function_derivative = lambda x: np.where(x > 0, 1, 0.01)

    

    # train the neural network
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = np.dot(self.wih, inputs)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        # 오차는 (실제 값 - 계산 값)
        output_errors = targets - final_outputs
        # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_errors = np.dot(self.who.T, output_errors)

        if self.act_fun == "Signoid":
            # 은닉 계층과 출력 계층 간의 가중치 업데이트
            self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
            # 입력 계층과 은닉 계층 간의 가충치 업데이트
            self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
            pass

        elif self.act_fun == "Relu":
            
            self.who += self.lr * np.dot((output_errors * self.activation_function_derivative(final_outputs)), np.transpose(hidden_outputs))
            
            self.wih += self.lr * np.dot((hidden_errors * self.activation_function_derivative(hidden_outputs)), np.transpose(inputs))
            
            pass


    # query the neural network
    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T

        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = np.dot(self.wih, inputs)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hideen layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
    

    def save_pt(self, ep_num, name):
        if not os.path.exists('pt_file'):
            os.makedirs('pt_file')
        np.savez("pt_file/" + name + f"_pt_ep{ep_num}.npz", wih=self.wih, who=self.who)


    def update_weight_by_ptfile(self, pt_path):
        loaded_data = np.load(pt_path)
        self.wih = loaded_data['wih']
        self.who = loaded_data['who']


# neural network class definition
class NeuralNetwork2:

    #initial the neural network
    def __init__(self, input, hidden1, hidden2, output, lr, act_fun):
        # set number of nodes in each input, hidden, output layer
        self.inodes = input
        self.hnodes1 = hidden1
        self.hnodes2 = hidden2
        self.onodes = output

        # learning rate
        self.lr = lr

        self.act_fun = act_fun

        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.wih2 = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.who = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))
        
        if self.act_fun == "Signoid":
            
            self.activation_function = lambda x: scipy.special.expit(x)
            
            self.inverse_activation_function = lambda x: scipy.special.logit(x)

        elif self.act_fun == "Relu":
            
            self.activation_function = lambda x: np.where(x > 0, x, 0.01 * x)
            
            self.inverse_activation_function = lambda x: np.where(x > 0, x, 100 * x)

            self.activation_function_derivative = lambda x: np.where(x > 0, 1, 0.01)
    

    # train the neural network
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T


        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs1 = np.dot(self.wih1, inputs)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs2 = np.dot(self.wih2, hidden_outputs1)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs2 = self.activation_function(hidden_inputs2)


        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs2)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        # 오차는 (실제 값 - 계산 값)
        output_errors = targets - final_outputs
        hidden_errors1 = np.dot(self.who.T, output_errors)
        hidden_errors2 = np.dot(self.wih2.T, hidden_errors1)

        if self.act_fun == "Signoid":
            # 은닉 계층과 출력 계층 간의 가중치 업데이트
            self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs2))
            self.wih2 += self.lr * np.dot((hidden_errors1 * hidden_outputs2 * (1.0- hidden_outputs2)), np.transpose(hidden_outputs1))
            self.wih1 += self.lr * np.dot((hidden_errors2 * hidden_outputs1 * (1.0- hidden_outputs1)), np.transpose(inputs))
            pass

        elif self.act_fun == "Relu":
            # 은닉 계층과 출력 계층 간의 가중치 업데이트
            self.who += self.lr * np.dot((output_errors * self.activation_function_derivative(final_outputs)), np.transpose(hidden_outputs2))
            self.wih2 += self.lr * np.dot((hidden_errors1 * self.activation_function_derivative(hidden_outputs2)), np.transpose(hidden_outputs1))
            self.wih1 += self.lr * np.dot((hidden_errors2 * self.activation_function_derivative(hidden_outputs1)), np.transpose(inputs))
            pass

    
    # query the neural network
    def query(self, inputs_list):
                
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T

        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs1 = np.dot(self.wih1, inputs)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs2 = np.dot(self.wih2, hidden_outputs1)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs2)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    def backquery(self, targets_list):
        
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)


        # calculate the signal out of the hidden layer
        hidden_outputs2 = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs2 -= np.min(hidden_outputs2)
        hidden_outputs2 /= np.max(hidden_outputs2)
        hidden_outputs2 *= 0.98
        hidden_outputs2 += 0.01
        # calculate the signal into the hideen layer
        hidden_inputs2 = self.inverse_activation_function(hidden_outputs2)


        # calculate the signal out of the hidden layer
        hidden_outputs1 = np.dot(self.wih2.T, hidden_inputs2)
        # scale them back to 0.01 to .99
        hidden_outputs1 -= np.min(hidden_outputs1)
        hidden_outputs1 /= np.max(hidden_outputs1)
        hidden_outputs1 *= 0.98
        hidden_outputs1 += 0.01
        # calculate the signal into the hideen layer
        hidden_inputs1 = self.inverse_activation_function(hidden_outputs1)

        
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih1.T, hidden_inputs1)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
    

    def save_pt(self, ep_num, name):
        if not os.path.exists('pt_file'):
            os.makedirs('pt_file')
        np.savez("pt_file/" + name + f"_pt_ep{ep_num}.npz", wih1=self.wih1, wih2=self.wih2, who=self.who)


    def update_weight_by_ptfile(self, pt_path):
        loaded_data = np.load(pt_path)
        self.wih1 = loaded_data['wih1']
        self.wih2 = loaded_data['wih2']
        self.who = loaded_data['who']


class Mnist():

    def __init__(self, input_nodes = 784, hidden_nodes = 200, hidden_nodes1 = 200, hidden_nodes2 = 200, output_nodes = 10, hidden_layer = 1, lr = 0.01, act_fun = "Signoid", pt_path = None):
        # number of input, hidden and output nodes
        self.train_cnt = 0

        if hidden_layer == 1:
            # create instance of neural network
            self.n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, lr, act_fun = act_fun)
        elif hidden_layer == 2:
            # create instance of neural network
            self.n = NeuralNetwork2(input_nodes, hidden_nodes1, hidden_nodes2, output_nodes, lr, act_fun = act_fun)

        # Train용 mnist 파일 open
        with gzip.open("/home/jimin/jss_prac/first-steps-of-neural-network/mnist_train.gz", 'r') as f:
            self.training_data_list = [x.decode('utf8').strip() for x in f.readlines()]
            f.close()

        # Test 용 mnist 파일 open
        with gzip.open("/home/jimin/jss_prac/first-steps-of-neural-network/mnist_test.gz", 'r') as f:
            self.test_data_list = [x.decode('utf8').strip() for x in f.readlines()]
            f.close()

        self.pt_path = pt_path


    def train(self, epochs, name):

        for e in range(epochs) :

            train_cnt = 0

            for record in self.training_data_list:
                
                # print("asdf",len(record))
                # asdf 5,0,0,0,0,0,0,0,0,0,0,0,0,0,....
                
                # 케로드를 쉽표에 의해 분리
                all_values = record.split(',')
                # 입력 값의 법위와 값 조정
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

                # print("asdf",len(inputs))
                # asdf 784

                # 결과 값 생성 (실제 값인 0.99 외에는 모두 0.01)
                targets = np.zeros(10) + 0.01
                # all_values[0]은 이 레코드에 대한 결과 값
                targets[int(all_values[0])] = 0.99

                self.n.train(inputs, targets)

                ## create rotated variations
                # rotated anticlockwise by 10 degrees
                inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
                self.n.train(inputs_plusx_img.reshape(784), targets)

                # rotated clockwise by 10 degrees
                inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
                self.n.train(inputs_minusx_img.reshape(784), targets)
                
                train_cnt += 1
                if(train_cnt % 100 == 0):
                    print("epochs", e+1, "train count :", train_cnt)
                pass

            self.n.save_pt(ep_num = e, name = name)
            
            pass


    def test(self):
        # 신경망의 성능의 지표가 되는 성적표를 아무 값도 가지지 않도록 초기화
        scorecard = []

        for record in self.test_data_list:

            all_values = record.split(',')
            
            correct_label = int(all_values[0])
            #print(correct_label,"correct label")
            # 입력 값의 범위와 값 조정
            inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
            
            self.n.update_weight_by_ptfile(self.pt_path)
            
            outputs = self.n.query(inputs)
            # 가장 높은 값의 인덱스는 레이블의 인덱스와 일치
            label = np.argmax(outputs)
            #print(label,"network's answer")
            
            if(label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
            pass

        # 정답의 비율인 성적을 계산해 출력
        scorecard_array = np.asarray(scorecard)
        
        performance = scorecard_array.sum()/scorecard_array.size

        print("performance = ",  performance, "total", scorecard_array.size)

        # label to test
        label = 0
        # create the output signals for this label
        targets = np.zeros(10) + 0.01
        # all_values[0] is the target label for this record
        targets[label] = 0.99
        print(targets)

        # get image data
        image_data = self.n.backquery(targets)

        # plot image data
        # image_array = image_data.reshape((28,28))
        # plt.imshow(image_array, cmap='Greys', interpolation=None)
        # plt.show()
        return performance


    def resume(self, additional_epoch):
       
        # ep 옆에 숫자 추출
        ep_num = re.search(r'ep(\d+)', self.pt_path)
        if ep_num:
            ep_num = int(ep_num.group(1))
        else:
            ep_num = None

        # _pt 왼쪽 이름 추출
        name = re.search(r'/([^/]+)_pt_', self.pt_path)
        if name:
            name = name.group(1)
        else:
            name = None

        print(name)


        self.n.update_weight_by_ptfile(self.pt_path)
        
        for e in range(additional_epoch) :

            train_cnt = 0

            for record in self.training_data_list:
                
                all_values = record.split(',')
                # 입력 값의 법위와 값 조정
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

                # print("asdf",len(inputs))
                # asdf 784

                # 결과 값 생성 (실제 값인 0.99 외에는 모두 0.01)
                targets = np.zeros(10) + 0.01
                # all_values[0]은 이 레코드에 대한 결과 값
                targets[int(all_values[0])] = 0.99

                self.n.train(inputs, targets)

                ## create rotated variations
                # rotated anticlockwise by 10 degrees
                inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
                self.n.train(inputs_plusx_img.reshape(784), targets)

                # rotated clockwise by 10 degrees
                inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
                self.n.train(inputs_minusx_img.reshape(784), targets)
                
                train_cnt += 1
                if(train_cnt % 100 == 0):
                    print("epochs", e+1, "train count :", train_cnt)
                pass
                
            self.n.save_pt(ep_num = ep_num + e + 1, name = name)
            
            pass


    def custum_img_test(self, img_path, mode = "white_background"):

        img = cv2.imread(img_path)

        gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

        if mode == "white_background":
            
            white_bg_img = np.ones((28,28))*255
            resized_img = cv2.resize(gray_img, (28,28))
            resized_img = white_bg_img - resized_img 
        
        if mode == "black_background":
            
            resized_img = cv2.resize(gray_img, (28,28))

        #print(resized_img)

        cv2.imwrite("test.png", resized_img)
        
        flattened_img = resized_img.reshape(-1)

        inputs = (np.asfarray(flattened_img)/255.0*0.99) + 0.01
        
        self.n.update_weight_by_ptfile(self.pt_path)
        
        outputs = self.n.query(inputs)

        label = np.argmax(outputs)

        print("label : ",label)


def main():

    # pt_file_path = "/home/jimin/jss_prac/first-steps-of-neural-network/pt_file/hidden_1000_pt_ep9.npz"
    # pt_file_path = "/home/jimin/jss_prac/first-steps-of-neural-network/pt_file/hidden_400_relu_pt_ep9.npz"
    pt_file_path = "/home/jimin/jss_prac/first-steps-of-neural-network/pt_file/hidden_1000_pt_ep9.npz"

    hidden_layer = 1

    hidden_nodes = 400
    hidden_nodes1 = 400
    hidden_nodes2 = 400

    lr = 0.01
    
    act_fun = "Signoid"

    if hidden_layer == 1:
        MNIST = Mnist(input_nodes = 784, hidden_nodes = hidden_nodes, output_nodes = 10, hidden_layer = hidden_layer, lr = lr, act_fun = act_fun, pt_path=pt_file_path)
    elif hidden_layer == 2:
        MNIST = Mnist(input_nodes = 784, hidden_nodes1 = hidden_nodes1, hidden_nodes2 = hidden_nodes2, output_nodes = 10, hidden_layer = hidden_layer, lr = lr, act_fun = act_fun, pt_path=pt_file_path)

    mode = "test"
    #mode = "train"
    #mode = "custum"
    #mode = "resume"

    if mode == "test":
        
        MNIST.test()
    
    elif mode == "custum":
        
        img_path = "/home/jimin/jss_prac/first-steps-of-neural-network/test_imgs/9.png"
        mode = "white_background"
        MNIST.custum_img_test(img_path=img_path, mode=mode)
    
    elif mode == "train":

        name = "hidden_400_lr_0.01"
        epochs = 10
        
        MNIST.train(epochs = epochs, name = name)

    elif mode == "resume":

        MNIST.resume(additional_epoch = 15)

def epochs_test():

    file_path = "/home/jimin/jss_prac/first-steps-of-neural-network/pt_file/hidden_400_pt_ep"

    hidden_layer = 1

    hidden_nodes = 400
    
    act_fun = "Signoid"

    epochs = []
    performances = []

    for i in range(35):

        pt_file_path = file_path + str(i) + ".npz"
        
        MNIST = Mnist(input_nodes = 784, hidden_nodes = hidden_nodes, output_nodes = 10, hidden_layer = hidden_layer, act_fun = act_fun, pt_path=pt_file_path)

        performance = MNIST.test()

        epochs.append(i+1)
        performances.append(performance)
    
    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, performances, marker='o', linestyle='-', color='b', label='performance')
    plt.title('Performance vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend()
    plt.show()

def hidden_node_test():

    file_path = "/home/jimin/jss_prac/first-steps-of-neural-network/pt_file/hidden_"

    hidden_layer = 1
    
    act_fun = "Signoid"

    epochs = []
    performances = []

    for i in range(9):

        hidden_nodes = (i+1)*100
        
        pt_file_path = file_path + str(hidden_nodes) + "_pt_ep9.npy"
        
        MNIST = Mnist(input_nodes = 784, hidden_nodes = hidden_nodes, output_nodes = 10, hidden_layer = hidden_layer, act_fun = act_fun, pt_path=pt_file_path)

        performance = MNIST.test()

        epochs.append(i+1)
        performances.append(performance)
    
    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, performances, marker='o', linestyle='-', color='b', label='performance')
    plt.title('Performance vs. Hidden_nodes')
    plt.xlabel('Hidden_nodes')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    hidden_node_test()
    #epochs_test()

