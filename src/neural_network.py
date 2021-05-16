# -*- coding: utf-8 -*-

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

a] 3 layers. Input, hidden, and output, 1 layer each.
b] conditions for the best number of perceptrons in hidden layer:
    	The number of hidden neurons should be between the size of the input layer and the size of the output layer.
    	The number of hidden neurons should be less than twice the size of the input layer.
    	citation: https://www.heatonresearch.com/2017/06/01/hidden-layers.html
        so our choice is 11.
  ouptut layer has 1 perceptron again.
c] batch processing is done in loop here. The weight matrix 1
   dimensions no.of features* no. of hidden layer neurons, i.e, 9*11 here
   weight matrtix 2 ha dimentions 11*1.
   bias is 0 , this combined with the activation functions used below gave us the best results among the options we had.
d] input to hidden has relu as activation, hiddden to output has sigmoid.
e] loss funtions is mean squared error. this can be seen in function backprop, where new_weights1 variable calculation
    has a term 2*(self.y - self.out) * sigmoid_derivative(self.out), which is the expression given by differentiating the
    loss function.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Activation functions


#leaky relu for hidden layer
def lrelu(t):
    temp=np.zeros(t.shape)
    for i in range(len(t)):
        for j in range(len(t[i])):
            temp[i][j]=max(0.01*t[i][j],t[i][j])
    return temp

# Derivative of relu
def lrelu_derivative(p):
    temp=np.zeros(p.shape)
    for i in range(len(p)):
        for j in range(len(p[i])):
            if p[i][j]==0:
                temp[i][j]=0
            else:
                temp[i][j]=1
    return temp

#sigmoid for output layer
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)



class NN:

    #''' X and Y are dataframes '''
    

    
    def feedforward(self):
        #leaky relu being the activation in step1 and sigmoid at the next one.
        self.layer1 = lrelu(np.dot(self.input, self.weights1))
        self.out = sigmoid(np.dot(self.layer1, self.weights2))
        return self.out
        

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        t1=np.transpose(self.layer1)
        t2=np.transpose(self.input)
        t3=np.transpose(self.weights2)
        
        #calculate all the change in weights
        #0.02= 0.01(learning rate)*2(this 2 is obtained during differentiating loss function, i.e, MSE here)
        new_weights_2 = np.dot(t1, (2*(self.y - self.out) * sigmoid_derivative(self.out)))
        new_weights_1 = np.dot(t2,  (np.dot(0.02*(self.y - self.out) * sigmoid_derivative(self.out), t3) * lrelu_derivative(self.layer1)))

       
        
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += new_weights_1
        self.weights2 += new_weights_2


    def fit(self,X,Y):
        #get dataframe X and Y as a numpy array object
        x=np.array(X,dtype=float)
        
        y=np.array(Y,dtype=float)
        y=y.reshape(y.shape[0],1)
        
        self.input=x
        self.y=y
        #initialization of weights
        np.random.seed(3)
        self.weights1=np.random.uniform(-20,20,(self.input.shape[1],11))
        self.weights2=np.random.uniform(-1,1,(11,1))
        
        self.weights1=self.weights1.tolist()
        self.weights2=self.weights2.tolist()
        
        self.out= np.zeros(self.y.shape)
        #training the batch in loop
        for i in range(1000):
            self.feedforward()
            self.backprop()
        return self.out



    def predict(self,X):

        x=np.array(X,dtype=float)
        self.input=x
        yhat=self.feedforward()
        for i in range(len(yhat)):
            if yhat[i]>=0.6:
                yhat[i]=1
            else:
                yhat[i]=0

        return yhat

    def CM(self,y_test,y_test_obs):# self is added
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        a=(tp+tn)/(tp+tn+fp+fn)
        print("  Confusion Matrix : ")
        print("  "+str(cm))
        print("\n")
        print(f"  Precision : {p}")
        print(f"  Recall : {r}")
        print(f"  F1 SCORE : {f1}")
        print(f"  Accuracy:{a}")

  
def main():
    #read  the processed csv file
    df=pd.read_csv('processed.csv')
    np.random.seed(4)
    #generate test train split
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=24)

    #creating 6 mini batches (6 being optimal for the dataset, by trial and error) 
    shuffled = X_train.sample(frac=1)
    result = np.array_split(shuffled, 6)

    np.random.seed(3)
    #pass it to fit and predict
    A=NN()
    i=1
    #training by sending each batch and at the end printing the performance metrics at the end of that mini batch training
    for df in result:
        y_train=df[df.columns[-1]]
        X=df.loc[:, df.columns !='Result' ]
        a=A.fit(X,y_train)
        print("")
        print("Training batch number ",i)
        print("") 
        i+=1
        y_train=np.array(y_train,dtype=float)
        y_train=y_train.reshape(y_train.shape[0],1) 
        A.CM(y_train,a)
    
    #testing chunk of code
    y_test=X_test[X_test.columns[-1]]
    X_test=X_test.loc[:, X_test.columns !='Result' ]
    y_test_obs=A.predict(X_test)
    
    #make y_test an np array and pass it to CM

    y_test=np.array(y_test,dtype=float)
    y_test=y_test.reshape(y_test.shape[0],1)
    print("")    
    print("In TESTING :")
    print("") 
    A.CM(y_test,y_test_obs)



#call main
main()

