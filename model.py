from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix 

# download and prep our training data
train_data = datasets.MNIST(
    root="data",        # where data will be stored
    train=True,         # get the training portion of mnist
    download=True,      # download it if not already
    transform=ToTensor()    # convert imeages to pytorch tensors
)

test_data = datasets.MNIST(
    root="data",
    train=False,            # get the testing portion (hence train = false)
    download=True,
    transform=ToTensor()
)

# check how many images are in each dataset
#print("Training images:", len(train_data))
#print("Testing images:", len(test_data))

# print out the first item of our data(color channel, pixels, pixels) then the written number
image, label = train_data[0]
#print("Image shape:", image.shape)
#print("Label:", label)

# loads our training images in batches of 64 and then shuffles them
test_loader = DataLoader(  test_data,  batch_size = 64 ,shuffle = False)
train_loader = DataLoader( train_data, batch_size = 64, shuffle = True)


class NeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # our first layer  784 pixels to 128 features
        self._layerOne = nn.Linear(784,128)
        # second layer is 128 features to 64 features
        self._layerTwo = nn.Linear(128,64)
        # third layer is 64 features to 10 possible digits
        self._layerThree = nn.Linear(64,10)
        # call our activation function
        self._ReLU = nn.ReLU()

    # takes an image and runs it thorugh the network
    def forward(self,x):

        # flatten our 28x28 image into 784 values
        x = torch.flatten(x,1)

        # pass our image through the first layer
        output_one = self._layerOne(x)

        # apply our relu
        output_one = self._ReLU(output_one)

        # pas through our secnond layer
        output_two = self._layerTwo(output_one)

        # apply our relu again
        output_two = self._ReLU(output_two)

        # pass through our final layer
        output_three = self._layerThree(output_two)

        # return the result
        return output_three

# create our neural network
model = NeuralNetwork()
# measure how wrong the models predictions were
loss_function = nn.CrossEntropyLoss()
# updates the models weights to improve its predictions
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
count = 0
# train our model for 10 epochs
while count < 10:

    total_loss = 0

    # go through training data one batch at a time
    for images, labels in train_loader:

        # clear our gradients from the previous batch
        optimizer.zero_grad()

        # make our predicitons
        result = model(images)

        # compare our predicitons to the correct answers
        loss = loss_function(result,labels)

        # call backprop to calculate how our weights should change
        loss.backward()

        # update our weights
        optimizer.step()

        # get the total loss from the current epoch
        total_loss += loss.item()

    #average it out so we get the average of the whole epoch
    average_loss = total_loss / len(train_loader)

    # print the loss after each epoch
    print(f"Epoch {count + 1}/10 - Average Loss: {average_loss}")
    count += 1

# now we test the model on images it has never seen

# counter variables
correct = 0
total = 0

# we don't want to train while testing of course!
all_labels = []
all_predictions = []
with torch.no_grad():

    for images, labels in test_loader:

        # make predictions
        result = model(images)

        # get the prediction with the highest score
        predictions = torch.argmax(result, dim=1)

        # append for our cm
        all_predictions.append(predictions)

        # count how many predictions were correct
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # append for our cm
        all_labels.append(labels)

# cat our lables for our confusion matrix
all_labels = torch.cat(all_labels)
all_predictions = torch.cat(all_predictions)
# calculate our accuracy
accuracy = correct / total

# print out our results
print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# check our accuracy on training data
correct_train = 0
total_train = 0

with torch.no_grad():

    # loop thorugh our training loader
    for images, labels in train_loader:
        result = model(images)
        predictions = torch.argmax(result, dim=1)

        # get our amount correct ofr training and its total
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

# calculate the accuracy
training_accuracy = correct_train / total_train

# print out our results
print("Training Accuracy:", training_accuracy)
print("Training Accuracy percentage:", training_accuracy * 100, "%")

# print out our confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print(cm)
