# Rupay Classifier



Abstract  
Advancement of Technology has replaced humans in almost every field with machines. By introducing machines, banking automation has reduced human workload. More care is required to handle currency, which is reduced by automation of banking. The identification of the currency value is hard when currency notes are blurry or damaged. Complex designs are included to enhance security of currency. This makes the task of currency recognition very difficult. To correctly recognize a currency it is very significant to choose the good features and suitable algorithm. In proposed method, Money Detector is used for segmentation and for classification, Deep learning is used which gives 95.6% accuracy. Foreign and Visually disable people in India often find difficulties in recognizing different currency notes. Even if some time it is also difficult for Indian healthy people to identify same amount of currency notes with different-new designs. Human eye has also some limitation so some time fake currency not identifiable by them. In this project using deep learning technique, detection model trained with dataset and tested it with different Indian currency with good accuracy.


Introduction
Currency is the paper notes and normal coins, which is releases for circulating within an economy by the government. It is the medium of switch for services and goods. For transaction, paper currency is an important medium. Characteristics of paper currency are simplicity, durability, complete control and cheapness. Due to this, it become popular. Among all other alternative forms of currency, the most preferable form of the currency is the paper. There is a one drawback of paper currencies which is that it cannot be reused but compared with the other methods this problem is not that much serious. As the part of the technological progression introduced to the banking sectors, financial institution and banking had started financial self-services. By using ATM counter and Coin - dispenser automated banking system is achieved where machines are used to handled currencies. In such situations, the machine will use the currency recognizer for the classification of the bank notes. Currency has two types of features internal and external features. External features include physical aspects of the currency like width and size. But these physical features are not reliable because currencies may damage due to circulation. Due to this damaged currencies system may fail to recognize currencies. Internal feature includes the Colour feature, which is also not reliable because currencies are passed through various hand and due to this, it becomes dirty which may give incorrect result. For currencies of each denomination there is a specific colour and size followed by Reserve bank of India. It is a very simple for human to identify the denomination of currency note because our brain is extremely skillfull in learning new matters and discovering them later without much trouble. But this currency recognition task turns very challenging in computer vision, in cases when currencies becomes damaged, old, and faded due to wear and tear. Security features are included in every Indian Currency which provides help in recognition and identification of the currency value.
Aim of this project is to propose a concept of Indian currency recognition. Thus, this project is to do this project for a blind person who cannot see anything, so this work can help blind people to recognize currency, they will have App/device for this, which can scan currency from camera image and predict the result and tell to the blind user which currency is this via voice. 

Working Structure
In this system, we have to just feed image of currency or video, which contain currency, it can detect which currency is this. So now question is that what mechanism is running in behind of this? The answer is very simple we are using Deep learning.

Dataset
Dataset is the most important thing in this project, it should be clear and very high quality. Quality of images matters a lot on our classification result, another thing is background, it should be clear in every images of prepared dataset otherwise model can get confused with objects & noise. So data gathering is the most important thing. Now 2nd thing is training, I am training my machine to learn to detect & recognize object from given input, so mine machine learning is based on what it see, and it is seeing and learnt from their image dataset.



Data gathering is the most important part of any research. In the 1st Notebook, I gathered dataset of new Indian currencies as well as old Indian Currencies from Kaggle. After image labelling, I created training and testing subfolder in IndianCurrency main folder, then we spilt our images into training & testing. After that, I created zip file for both training & testing, it contains image label, which we selected during image labelling, and we did this with our python code and upload it on Kaggle  . Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like. The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the ‘fastai’ library has a handy function made exactly for this, ImageDataBunch.from_name_re gets the labels from the filenames using a regular expression.
I define what will be our label name & what will be our image count for particular one currency, compress the images in zip.. So now first step is done.

Deep Learning
Deep learning is becoming the most evolving AI technique in 21st century. In this modern era, we have lots of data, and deep learning needs a data, that’s why it becomes so popular in this modern era. We can see the usage of deep learning everywhere like social media, government, IT sector, Cinemas, Search engines. We humans already implemented Face recognition, self-driving car, auto drones and we are continuously evolving like we are in infinity loop of mega evolution of AI. Now let us understand some algorithms that we used in our project. 



Convolutional Neural Network
A Convolutional Neural Network is most popular Deep learning algorithm in which it takes an input image, assign weights and biases to various aspect according to the object in the image. In other algorithms we need to do lots of image processing and hand engineering to achieve the accuracy. But in CNN have the ability to learn these all the characteristics of images. So, we don’t need to do a lots of hand engineering in images, CNN will do for us. And also, we can achieve a good accuracy in our work.
Working of Convolutional Neural Network (CNN)
A Convolutional Neural Network have a n numbers of layers which can learn to detect different features from an image data, and the output of each processed image is used as the input to the next layer. The filters or we can say processing like edges, increase complexity, adjust brightness. CNN can perform feature identification classification of images, sound, audio, video and text [2].
CNN is composed of an input layer, an output layer, and many hidden layers in network.
These layers perform learning operation on the given data, Convolution function, Activation function and pooling are hidden layers (Fig. 3).
 

Layers of CNN
 
•	Convolution It have set of convolutional filters, which find features from images, so images pass through these all filters .

•	Rectified linear unit (ReLU) is useful for mapping negative values into zero so it’s maintaining positives values, so this is one kind of activation function, we have many more choices like Sigmoid, hyperbolic tangent, but choosing a layer for a model is a depends on your data. It affects the accuracy.
•	Pooling performs the non-linear down stamping which can reduce the number of parameters then the network needs to learn and simplify the output.

•	Dense layer is collections of neurons. It describes how neurons connected to the next layer of neurons (In short each neuron is connected to every neuron in the next layer). It is also known as Fully Connected layer [8].
These operations iterative on neural network layers, in which each layer learning to identify different features.

Training Time Increasing with GPU
A convolutional neural network is trained on hundreds, thousands, or even millions of images. When we have to work with lots of data then we can use GPUs for processing and computing. It can decrease the model training time and after training we can use our model in real world application.
Tools and Technology Used
Python Programming Language.
Python is programming language like C, C++, C#. It is an interpreted high-level programming language. Guido van Rossum created it, and it was first released in 1991. Python coding style is so comfortable for programmer, it has indentation feature so our code structure always stays good and understandable for other. Python is dynamically type language and also, it’s have garbage collection so we don’t feel to worry about unnecessary garbage in programming, It supported Procedural programming, functional programming and also object oriented programming.
Experimental Approach
During this project, Resnet50 model is used for training. Let us understand about it and pre-trained model.
Pre-trained Model
A pre-trained model, a name defining this term, it means model is trained on large dataset. You can use directly pre-trained model, in this just you have to feed your data and it can train on your data but it is already learnt on the large dataset, now you are re-training model so it will give you a better result. This learning approach is called Transfer learning. For example, you trained network on one lakhs images and now you are retraining it on 500 images for a classification purpose. There are some Transfer learning models like Googlenet, Imagenet, Alexanet, VGG16, VGG19, RCNN Inception and many more models you can use from tensorflow, keras and pytorch.
Workflow
In 2nd Notebook, I need to define our deep learning model. So we have downloaded a faster resnet50 model. It is already pre-trained model just we need to feed our images. After that, we need to edit config file of model and need to give a path of our images, record file. Then I did training in II notebook (Train.ipynb) which is available in aiprojects folder in drive. Our training taken almost 1/2h, after the getting expected accuracy, I stopped the training. With code, we generate the confusion matrix of our training. Resnet50 (Pre-Trained model) gives me 0.95 accuracy. Now it is time for testing. 







 
                                         ACCURACY OF PROPOSED SYSTEM



                                                        Confusion Matrix


Results
In 3rd Notebook, I tested our models on different currencies. You can see the results & testing below.
  

     
     
    Images shows output of our trained model for different amount notes.

Conclusion
By learning from Geeksman AI squad on Image Classification and Deep learning, I in this project choose resnet50 model to train model and recognize Indian currency very well, which will help a lot to visual disable, foreign and old age people.
In future work planning to increase accuracy of currency recognition.
We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly. Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

References
www.kaggle.com
www.stackoverflow.com
https://colab.research.google.com/github/pttrilok/practical_deep_learning/blob/master/Lesson%201%20Computer%20Vision.ipynb#scrollTo=sUe9Wxmh01iv
https://colab.research.google.com/github/pttrilok/practical_deep_learning/blob/master/Lesson%202%20Computer%20Vision.ipynb
https://colab.research.google.com/github/pttrilok/practical_deep_learning/blob/master/Lesson%203%20Computer%20Vision.ipynb#scrollTo=uNGsyTjC7Apf





