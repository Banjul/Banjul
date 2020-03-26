# Banjul
Question classifiers

[OS]
MACOS

[Project architecture]

[data] folder:
Under this folder, there are three files including dev, train and test data. 
If you want to run the model using pre-trained word embedding, please put the file [glove.small.txt] under this folder. Or if you use other name please change the value of key [pre_trained_path] in config file to the corresponding name. 
After testing the model, an output.txt file will be saved under this folder.

[src] folder (i.e. root directory)
1. model.py: construction of BOW and biLSTM models
2. dataLoad.py:  data processing procedures. Include preprocessing and word to index and so on. 
3. question_classifier.py:  execution of the question classifier.
4. bow_config.config
5. biLSTM_config.config
6. bow.mdl/biLSTM.mdl: after the training process, the model will be saved under this directory.


[Config parameters]
Open bow_config.config or biLSTM_config.config file and change the values of keys for certain purposes.

For using pre-trained word embeddings: use_pre_trained = True
For using randomly initial word embeddings: use_pre_trained = 

For using freeze: freeze = True
For using fine tuning: freeze =

[Interface]
Attention: please do not delete any file in under src folder.

For training bow model run, cd to the project directory: 
% python3 question_classifier.py train --config bow_config.config
For testing bow model run:
% python3 question_classifier.py test --config bow_config.config


For training biLSTM model run, cd to the project directory: 
% python3 question_classifier.py train --config biLSTM_config.config
For testing biLSTM model run:
% python3 question_classifier.py test --config biLSTM_config.config


[Output file]
After testing the trained model, the testing results will be written in ./data/output_file.txt. 
It includes the predicted and real labels for each question in test.txt. 
After these, test_time, test_loss, test_accuracy, amount of rightly predicted questions and total number of questions will also be documented in this file.
The evaluate results of the second trained model will be appended, which means the first tested information will not be overwritten.
