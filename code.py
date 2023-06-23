import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#reading the csv data file
dataset=pd.read_csv("kc_house_data.csv")
X=dataset[['bedrooms','bathrooms','sqft_living','lat','long']].values
Y=dataset[['price']].values

#splitting the dataset into train(75%) and test(25%) samples
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

#applying data cleaning i.e., preprocessing
scaler= StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#creating artificial neural network

#varying batch size
batch_size=[5,10,15,20]
acc=[]
for x in batch_size:
  model=Sequential()
  model.add(Dense(units=5,input_dim=X_train.shape[1],kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(units=5,kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='sgd')
  model.fit(X_train,Y_train,batch_size=x,epochs=50)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('BatchSize:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(batch_size,acc)
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying batch sizes')
plt.show()

#varying epochs
epochs=[25,50,100]
acc=[]
for x in epochs:
  model=Sequential()
  model.add(Dense(units=5,input_dim=X_train.shape[1],kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(units=5,kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='sgd')
  model.fit(X_train,Y_train,batch_size=10,epochs=x)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('Epochs:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(epochs,acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying epochs')
plt.show()

#varying activation function
activation=['relu','sigmoid','tanh']
acc=[]
for x in activation:
  model=Sequential()
  model.add(Dense(units=5,input_dim=X_train.shape[1],kernel_initializer='normal',activation=x))
  model.add(Dense(units=5,kernel_initializer='normal',activation=x))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='sgd')
  model.fit(X_train,Y_train,batch_size=10,epochs=50)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('Activation Function:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(activation,acc)
plt.xlabel('Activation Function')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying activation')
plt.show()

#varying optimizer
optimizer=['sgd','adam','rmsprop']
acc=[]
for x in optimizer:
  model=Sequential()
  model.add(Dense(units=5,input_dim=X_train.shape[1],kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(units=5,kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer=x)
  model.fit(X_train,Y_train,batch_size=10,epochs=50)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('Optimizer:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(optimizer,acc)
plt.xlabel('Optimizer')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying optimizer')
plt.show()

#varying loss function
loss=['mean_absolute_error','cosine_similarity','mean_squared_error']
acc=[]
for x in loss:
  model=Sequential()
  model.add(Dense(units=5,input_dim=X_train.shape[1],kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(units=5,kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss=x,optimizer='sgd')
  model.fit(X_train,Y_train,batch_size=10,epochs=50)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('Loss Function:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(loss,acc)
plt.xlabel('Loss Function')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying loss function')
plt.show()

#varying nodes
nodes=[3,5,10]
acc=[]
for x in nodes:
  model=Sequential()
  model.add(Dense(units=x,input_dim=X_train.shape[1],kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(units=x,kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='sgd')
  model.fit(X_train,Y_train,batch_size=10,epochs=50)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('Nodes:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(nodes,acc)
plt.xlabel('Nodes')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying nodes')
plt.show()

#varying hidden layers
layers=[2,5,7]
acc=[]
for x in layers:
  model=Sequential()
  model.add(Dense(units=10,input_dim=X_train.shape[1],kernel_initializer='normal',activation='sigmoid'))
  for _ in range(x-2):
    model.add(Dense(units=10,kernel_initializer='normal',activation='sigmoid'))
  model.add(Dense(1,kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='sgd')
  model.fit(X_train,Y_train,batch_size=10,epochs=50)
  MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
  acc.append(100-MAPE)
  print('Hidden Layers:',x,' Accuracy:',100-MAPE)
#comparing accuracies 
plt.plot(layers,acc)
plt.xlabel('Hidden Layers')
plt.ylabel('Accuracy')
plt.title('Comparing accuracies for varying layers')
plt.show()

#finding best hyperparameters for ANN using manual grid search
def BestHypPar(X_train,Y_train,X_test,Y_test):
    lr=[]
    batch_size=[5,10,15,20]
    epoch=[25,50,100]
    activation=['relu','sigmoid']
    optimizer=['sgd','adam','rmsprop']
    loss_func=['mean_squared_error','mean_absolute_error','cosine_similarity']
    hidden_layers=[2,5,7]
    nodes=[3,5,10]
    Results=pd.DataFrame(columns=['Trial','Parameters','Accuracy'])
    Trial=0
    for a in batch_size:
        for b in epoch:
            for c in activation:
                for d in optimizer:
                    for e in loss_func:
                        for f in hidden_layers:
                            for g in nodes:
                                Trial+=1
                                model=Sequential()
                                model.add(Dense(units=g,input_dim=X_train.shape[1],kernel_initializer='normal',activation=c))
                                for _ in range(f-2):
                                    model.add(Dense(units=g,kernel_initializer='normal',activation=c))
                                model.add(Dense(1,kernel_initializer='normal'))
                                model.compile(loss=e,optimizer=d)
                                model.fit(X_train,Y_train,batch_size=a,epochs=b)
                                MAPE=np.mean(100*(np.abs(Y_test-model.predict(X_test))/Y_test))
                                print('Hyperparameters',Trial,': ','BatchSize:',a,' Epochs:',b,' ActivationFunction:',c,' Optimizer:',d,' LossFunction:',e,' HiddenLayers:',f,' Nodes:',g,' Accuracy:',100-MAPE)
                                Results=Results.append(pd.DataFrame(data=[[Trial,str(a)+'-'+str(b)+'-'+str(c)+'-'+str(d)+'-'+str(e)+'-'+str(f)+'-'+str(g),100-MAPE]],columns=['Trial','Parameters','Accuracy']))
    return(Results)

Output=BestHypPar(X_train,Y_train,X_test,Y_test)
print(Output)
