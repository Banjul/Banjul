import torch
import argparse
import configparser
from dataLoad import Corpus
from dataLoad import DataLoader
from dataLoad import DataSplit
from dataLoad import dataProcessor
import time
import model

parser = argparse.ArgumentParser()
parser.add_argument("testOrTrain", type = str, help = "test or train the model")
parser.add_argument("--config", type=str, help="configuration file path", required=True)

args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

#choose model
model_config = config['Model']
model_to_use = str(model_config['model'])

#output path
output_config = config['Evaluation']
path_eval_result = str(output_config['path_eval_result'])


#choose embedding
embedding_config = config['Embedding']
pre_trained = bool(str(embedding_config['use_pre_trained']))
pre_trained_path = str(embedding_config['pre_trained_path'])

criterion = torch.nn.CrossEntropyLoss()

if model_to_use == 'biLSTM':
    #traing parameters
    training_config = config['Training']
    lr = float(training_config['lr'])
    batch_size = int(training_config['batch_size'])
    seed = int(training_config['seed'])
    dropout = float(training_config['dropout'])
    embed_dim = int(training_config['embed_dim'])
    hidden_size = int(training_config['hidden_size'])
    bidirectional = bool(str(training_config['bidirectional']))
    weight_decay = float(training_config['weight_decay'])
    sequence_length = int(training_config['sequence_length'])
    freeze = bool(training_config['freeze'])
    epochs = int(training_config['biLSTM_epochs'])
    model_save = str(model_config['bilstm_path_model'])
    torch.manual_seed(seed)

    
    #load data
    file_path = "./data/"
    save_data = "./corpus"
    max_lenth = 16
    corpus = Corpus(file_path, save_data, max_lenth)
    corpus.save()
    data = save_data
    data = torch.load(data)
    max_len = data["max_len"]
    vocab_size = data['dict']['vocab_size']
    output_size = data['dict']['label_size']
    training_data = DataLoader(data['train']['src'],
                           data['train']['label'],
                           max_len,
                           batch_size=batch_size)
    validation_data = DataLoader(data['dev']['src'],
                             data['dev']['label'],
                             max_len,
                             batch_size=batch_size,
                             shuffle=False)
    test_data = DataLoader(data['test']['src'],
                             data['test']['label'],
                             max_len,
                             batch_size=batch_size,
                             shuffle=False)
    test_labels_words = data['test']['label_words']
    
    ###################################################
    #training

    train_loss = []
    valid_loss = []
    test_loss =[]
    valid_accuracy = []
    test_accuracy = []

    

    def biLSTM_evaluate():
        lstm_attn.eval()
        corrects = eval_loss = 0
        _size = validation_data.sents_size

        for data, label in validation_data:
            pred = lstm_attn(data)
            loss = criterion(pred, label)

            eval_loss += loss.data
            corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
        return eval_loss.data/_size, corrects, corrects*100.0/_size, _size

    def biLSTM_train():
        lstm_attn.train()
        total_loss = 0
        for data, label in training_data:
            optimizer.zero_grad()
            target = lstm_attn(data)
            loss = criterion(target, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        return total_loss.data/training_data.sents_size

    def biLSTM_test(trainedModel,test_labels_words,output_file,label_table):
        trainedModel.eval()
        corrects = eval_loss = 0
        _size = test_data.sents_size
        predictions =[]
        output_file.write('-' * 90 + '\n')
        output_file.write('biLSTM model test questions predict results:' + '\n')
        for data, label in test_data:
            pred = trainedModel(data)
            loss = criterion(pred, label)
            eval_loss += loss.data
            pred = torch.max(pred, 1)[1].view(label.size()).data
            for l in range(len(label)):
                predictions.append(pred[l].data)
            corrects += (pred == label.data).sum()
        rightness_test(predictions,test_labels_words,output_file,label_table)
        return eval_loss.data/_size, corrects, corrects*100.0/_size, _size

    def rightness_test(predictions, test_labels_words,output_file,label_table):
        wrong_labels = {}
        total_wrong_labels = 0
        for i in range(0,len(predictions)):
            pred = list(label_table.keys())[list(label_table.values()).index(predictions[i])]
            right = test_labels_words[i]
            output_file.write('Question ' + str(i+1) + 
                              ': Predict lable[ '   + pred + 
                              ' ] Right label [ ' + right + ' ]' + '\n' )
            if pred != right:
                total_wrong_labels += 1
                if pred not in wrong_labels:
                    wrong_labels[pred] = 1
                else:
                    wrong_labels[pred] += 1
        wrong_labels = sorted(wrong_labels.items(), key=lambda x: x[1])
            
    
    best_acc = None
    total_start_time = time.time()

    try:
        print('-' * 90)
        if args.testOrTrain == "train":
            lstm_attn = model.bilstm(batch_size=batch_size,
                                  output_size=output_size,
                                  hidden_size=hidden_size,
                                  vocab_size=vocab_size,
                                  embed_dim=embed_dim,
                                  bidirectional=bidirectional,
                                  dropout=dropout,
                                  sequence_length=sequence_length,
                                  pre_trained=pre_trained,
                                  pre_trained_path= pre_trained_path,
                                  freeze = freeze)


            optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=lr, weight_decay=weight_decay)

            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
        
                loss = biLSTM_train()
                train_loss.append(loss*1000.)

                print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                              loss))

                loss, corrects, acc, size = biLSTM_evaluate()
                valid_loss.append(loss*1000.)
                valid_accuracy.append(acc)

                print('-' * 10)
                print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | dev_accuracy {:.4f}%({}/{})'.format(epoch,
                                                                                                 time.time() - epoch_start_time,
                                                                                                 loss,
                                                                                                 acc,
                                                                                                 corrects,
                                                                                                 size))
    
                torch.save(lstm_attn, model_save)
        else:
            epoch_start_time = time.time()
            trainedModel = torch.load(model_save)
            output_file = open(path_eval_result,'a')
            label_table = data['dict']['label']
            loss, corrects, acc, size = biLSTM_test(trainedModel,test_labels_words,output_file,label_table)
            test_loss.append(loss*1000.)
            test_accuracy.append(acc)
            
            accuracy_string = '| test_time: {:2.2f}s | test_loss {:.4f} | test_accuracy {:.4f}%({}/{})'.format(
                time.time() - epoch_start_time,loss,acc,corrects,size)
            output_file.write('biLSTM model test evaluating result:' + '\n')
            output_file.write(accuracy_string + '\n')
            print(accuracy_string)

     
    except KeyboardInterrupt:
        print("-"*90)
        print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

elif model_to_use == 'bow':
    model_save = str(model_config['bow_path_model'])
    dl = DataSplit()
    
    # file path
    file_config = config['Files']
    train_filename = str(file_config['path_train'])
    dev_filename = str(file_config['path_dev'])
    test_filename = str(file_config['path_test'])
    train_data, train_sentences, train_all_words, train_labels = dl.load_split(train_filename)
    dev_data, dev_sentences, dev_all_words, dev_labels = dl.load_split(dev_filename)
    test_data, test_sentences, test_all_words, test_labels = dl.load_split(test_filename)

    dp = dataProcessor()
    train_word_to_ix = dp.word2index(train_data)
    train_label = dp.count_labels(train_labels)
    train_label_to_ix = dp.label2index(train_label)
    train_diction = dp.freq_dic(train_all_words)

    NUM_LABELS = len(train_label)
    VOCAB_SIZE = len(train_word_to_ix)

    def train_pro():
        x_train = []
        y_train = []
        for row in train_data:
            x_train.append(row[0])
            y_train.append(row[1])
        for i in range(len(y_train)):
            y_train[i] = train_label_to_ix[y_train[i]]

        y_train = torch.tensor(y_train)
        return x_train,y_train

    def dev_pro():
        x_dev = []
        y_dev = []
        for row in dev_data:
            x_dev.append(row[0])
            y_dev.append(row[1])
        for i in range(len(y_dev)):
            y_dev[i] = train_label_to_ix[y_dev[i]]
        y_dev = torch.tensor(y_dev)
        return x_dev,y_dev
    
    def test_pro():
        x_test = []
        y_test = []
        for row in test_data:
            x_test.append(row[0])
            y_test.append(row[1])
        for i in range(len(y_test)):
            y_test[i] = train_label_to_ix[y_test[i]]
        y_test = torch.tensor(y_test)
        return x_test,y_test

    losses = []
    
    def bow_train(x, y, model, cost,test_word_to_ix):
        optimizer.zero_grad()
        predict = model.forward(x, model.lookuptable,test_word_to_ix)
        right, total = rightness(predict, y)  # 返回值为（正确样例数，总样本数）
        loss = cost(predict, y)
        # 将损失函数数值加入到列表中
        losses.append(loss.data.numpy())
        # 开始进行梯度反传
        loss.backward()
        # 开始对参数进行一步优化
        optimizer.step()
        return right, total, loss

    def bow_evaluate(x, y, model, cost,test_word_to_ix):
        predict = model.forward(x, model.lookuptable,test_word_to_ix)
        right, total = rightness(predict, y)  # 返回值为（正确样例数，总样本数）
        loss = cost(predict, y)
        return right, total, loss
    
    def bow_test(x, y, model, cost, test_word_to_ix, test_labels,output_file):
        predict = model.forward(x, model.lookuptable, test_word_to_ix)
        
        right, total,all_pre_label = rightness_test(predict, y, test_labels)  # 返回值为（正确样例数，总样本数）
        output_file.write('-' * 90 + '\n')
        output_file.write('BOW model test questions predict results:' + '\n')
        for i in range(1,501):
            output_file.write('Question ' + str(i) + 
                              ': Predict lable[ '   + all_pre_label[i-1]+ 
                              ' ] Right label [ ' + test_labels[i-1] + ' ]' + '\n' )
        
        loss = cost(predict, y)
        return right, total, loss

    def rightness(predictions, labels):

        """
        计算预测错误率的函数，其中predictions是模型给出的一组预测结果，
        batch_size行num_classes列的矩阵，labels是数据之中的正确答案
        """
        pred = torch.max(predictions.data, 1)[1]
        # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标 ？？？？？
        rights = pred.eq(labels.data.view_as(pred)).sum()
        # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
        return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素

    def rightness_test(predictions, labels, y_test_label):

        pred = torch.max(predictions.data, 1)[1]
        # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标 ？？？？？
        rights = pred.eq(labels.data.view_as(pred)).sum()
        # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量

        ##############   以下为对错误样本的统计   ##############

        compare = pred.eq(labels.data.view_as(pred)).numpy()
        wrong_sample_index = []    #错误样本的下标
        for index, c in enumerate(compare):
            if c == False:
                wrong_sample_index.append(index)

        wrong_labels = []          #错误样本的正确标签
        for wrong_index in wrong_sample_index:
            wrong_labels.append(y_test_label[wrong_index])


        wrong_label_times = {}
        for key in wrong_labels:
            wrong_label_times[key] = wrong_label_times.get(key, 0) + 1

        pre_label_ix = pred.numpy()  #所有预测标签(index)
        pred_label_ix = [] 
        for i in wrong_sample_index:
            pred_label_ix.append(pre_label_ix[i])
        
        pred_label = []          #错样本的预测标签(label)
        for p in pred_label_ix:
            pred_label.append(list(train_label_to_ix.keys())[list(train_label_to_ix.values()).index(p)])

        right_wrong = []  #正确标签 -> 错误的预测标签
        for r in range(0, len(wrong_labels)):
            right_wrong.append(wrong_labels[r] +" -> "+ pred_label[r])
        
        all_pre_label = [] 
        for ix in pre_label_ix:
            all_pre_label.append(list(train_label_to_ix.keys())[list(train_label_to_ix.values()).index(ix)])

            
        return rights, len(labels), all_pre_label # 返回正确的数量和这一次一共比较了多少元素

    x_train,y_train = train_pro()
    x_dev, y_dev=dev_pro()
    x_test, y_test = test_pro()

    total_start_time = time.time()

    try:
        print('-' * 90)
        if args.testOrTrain == "train":
            #traing parameters
            training_config = config['Training']
            lr = float(training_config['lr'])
            seed = int(training_config['seed'])
            embed_dim = int(training_config['embed_dim'])
            hidden_size = int(training_config['hidden_size'])
            freeze = bool(training_config['freeze'])
            epochs = int(training_config['bow_epochs'])
            weight_decay = float(training_config['weight_decay'])

            torch.manual_seed(seed)
            
            m = model.bow_nn(input_size = embed_dim,
                    hidden_size = hidden_size,
                    output_size=NUM_LABELS,
                    emb_dim= VOCAB_SIZE,
                    pre_trained=pre_trained,
                    pre_trained_path= pre_trained_path,
                    freeze = freeze)
            
            optimizer = torch.optim.Adam(m.parameters(), lr=lr,weight_decay=weight_decay)
            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
                corrects, size, loss = bow_train(x_train, y_train, m, criterion,train_word_to_ix)
                acc = (1.0*corrects/size)*100
                print('| start of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | train_accuracy {:.4f}%({}/{})'.format(epoch,
                                                                                                   time.time() - epoch_start_time,
                                                                                                   loss,
                                                                                                   acc,
                                                                                                   corrects,
                                                                                                   size))

                corrects, size, loss = bow_evaluate(x_dev, y_dev, m, criterion,train_word_to_ix)
                acc = (1.0*corrects/size)*100
                print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | dev_accuracy {:.4f}%({}/{})'.format(epoch,
                                                                                                   time.time() - epoch_start_time,
                                                                                                   loss,
                                                                                                   acc,
                                                                                                   corrects,
                                                                                                   size))

                torch.save(m, model_save)
        else:
            epoch_start_time = time.time()
            trainedModel = torch.load(model_save)
            output_file = open(path_eval_result,'a')
            corrects, size, loss = bow_test(x_test, y_test, trainedModel, criterion,train_word_to_ix, test_labels,output_file)
            acc = (1.0 * corrects / size) * 100
            
            accuracy_string = '| test_time: {:2.2f}s | test_loss {:.4f} | test_accuracy {:.4f}%({}/{})'.format(
                time.time() - epoch_start_time,loss,acc,corrects,size)
            output_file.write('BOW model test evaluating result:' + '\n')
            output_file.write(accuracy_string + '\n')
            print(accuracy_string)
                                

    except KeyboardInterrupt:
        print("-"*90)
        print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

else: 
    print('Wrong model name')







