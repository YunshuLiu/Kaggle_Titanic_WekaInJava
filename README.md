Kaggle_Titanic_WekaInJava
=========================
1. This repository contains my Java code for the Titanic project in Kaggle competition. In order to run the code yourself, you will have to preprocess the data you downloaded from Kaggle-Titanic website and convert it to .arff format, the name of the training file is "trainV4_rmParch.arff", name of the test file is "test_V4_rmParch.arff". Note I also did some preprocessing before I feed them into Weka, for example, I removed feature "parch", extract lastname from feature "name", these preprocessing are proved to improve the performance of the prediction. In training dataset, the first column is the lable, if you put the label in other columns, you will need to modify "trainIns.setClassIndex(0)", for example, if your lable is in the last column, you can change the code to trainIns.setClassIndex(trainIns.numAttributes()-1).
2. My code used two libraries, one is Weka, which you can download at http://www.cs.waikato.ac.nz/ml/weka/, the other is Amten's fast Neural Network implementation, which you can download at https://github.com/amten/NeuralNetwork, in order to run my code, you will need to include weka.jar from Weka and NeuralNetwork.jar from Amten's repository in you path.
3. Functitality of each file:
 * TitanicSMO.java: implemention of SVM with PolyKernel or RBFKernel
 * TitanicSMOCVPS.java: implemention of SVM with PolyKernel, where you can setup a range of C paramenter using CVParameterSelection
 * TitanicSMOGridSearch.java: implemention of SVM with RBFKernel, where you can setup a grid for parameter C and kernel.gamma
 * TitanicSMOVoting.java: You can setup a ensemble algorithm which build several(in my case, five) models from same/different algorithms and use different rules(majority voting, average probability etc.).
 * TitanicWekaNN.java: use Amten's Neural network Weka plugin to build a neural network, I implement both fully connected neural network with network and convolutional neural network. From my experience, in both cases, Stochastic Gradient Descent outperforms mini-batch Gradient Descent. 
 * TitanicWekaNNVoting.java: voting with neural network.
4. Performance:
 * I get the best result 0.836 on Kaggle leader board from a fully connected neural network, the network I build contains two hidden layers, each have 500 nodes, the input dropout is set to 0.2, hidden layer dropout rate is 0.5.
 * The second best result 0.823 on Kaggle leader board from voting by seven different algorithms.
 * Besides Neural network, SVM give the best performance, both SMO with PolyKernel and with RBFKernel give very good performance. 
