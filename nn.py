import numpy as np 
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformation,padding
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from keras import metrics
import os
from sklearn.metrics import classification_report
from keras import backend as K
import tensorflow as tf

ROOTDIR = os.path.dirname(os.path.realpath(__file__))

def get_word_embeddings(filepath, words):
    """
    get file of token embeddings
    """
    fEmbeddings = open(filepath, encoding="utf-8")
    word2Idx = {}
    wordEmbeddings = []

    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]

        if len(word2Idx) == 0: # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)

    return word2Idx, np.array(wordEmbeddings)


def tag_dataset(model, dataset):
    """
    """
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):    
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing, char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i+1)
    return predLabels, correctLabels

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class CustomMetrics:

    def __init__(self, label2Idx):
        self.label2Idx = label2Idx

    def person_precision(self, y_true, y_predictions):
        i_per_idxs = tf.where(tf.reshape(y_true, shape=[-1])==self.label2Idx.get("I-PER"))
        print(i_per_idxs)

        predictions_persons = tf.reshape(tf.math.argmax(y_predictions, axis=2),shape=[-1])[i_per_idxs]
        labels_persons = y_true[i_per_idxs]
        return sum(labels_persons)
#        return precision_m(labels_persons, predictions_persons).astype("int64")

def get_model(model_number, wordEmbeddings, caseEmbeddings, char2Idx, label2Idx):
    """
    """

    words_input = Input(shape=(None,),dtype='int32',name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)

    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)

    character_input=Input(shape=(None,52,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    dropout= Dropout(0.5)(embed_char_out)
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
    maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)

    output = concatenate([words, casing, char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    print(label2Idx.get("I-PER"))
    #PersonPrecision
#    custom_metrics = CustomMetrics(label2Idx)
    model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='nadam',
#            metrics=[metrics.Precision(), metrics.Recall()],
            run_eagerly=True
    )
    model.summary()
    # plot_model(model, to_file='model.png')
    return model

def train_model(model, epochs, train_batch_len, train_batch, dev_batch_len, dev_batch, idx2Label, label2Idx):
    """
    train model for n epochs
    """
    history = {}
    for epoch in range(1, epochs):

        print("\nTrain Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(train_batch_len))

        for i, batch in enumerate(iterate_minibatches(train_batch, train_batch_len)):
            labels, tokens, casing, char = batch

            train_batch_results = model.train_on_batch([tokens, casing, char], labels, return_dict=True)

            a.update(i, values=train_batch_results.items())

        a.update(i+1)


        # track dev results
        dev_per = []

        print("Dev Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(dev_batch_len))

        for i, batch in enumerate(iterate_minibatches(dev_batch, dev_batch_len)):
            #   Performance on dev dataset
            labels, tokens, casing, char = batch

            test_batch_results = model.test_on_batch([tokens, casing, char], labels, return_dict=True)

            predictions = model.predict([tokens, casing, char], workers=10, use_multiprocessing=True)
            predictions = np.argmax(predictions, axis=2).flatten()

#            not_o_idxs = np.where(labels.flatten()!=label2Idx.get("O"))[0]
            i_per_idxs = np.where(labels.flatten()==label2Idx.get("I-PER"))[0]

#            dev_ents.append([labels.flatten()[not_o_idxs], predictions[not_o_idxs]])
            dev_per.append([labels.flatten(), predictions])

            a.update(i, values=test_batch_results.items())

        a.update(i+1)

        ground_truth = list(np.concatenate(np.array(dev_per, dtype=object)[:,0]))
        predictions = list(np.concatenate(np.array(dev_per, dtype=object)[:,1]))

        metrics_returned = classification_report(ground_truth, predictions, digits=4, zero_division=True, output_dict=True)
        print(metrics_returned.get("weighted avg").items())
#        print( "{} {:.4f} {} {:.4f} {} {:4f} {} {}".format(*i for i in ))
#        print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (precision_per, recall_per, f1_score_per))

    return model


def main(epochs, glove_embeddings_path):

    trainSentences = readfile(os.path.join(ROOTDIR, "data", "train.txt"))
    devSentences = readfile(os.path.join(ROOTDIR, "data", "valid.txt"))
    testSentences = readfile(os.path.join(ROOTDIR, "data", "test.txt"))

    trainSentences = addCharInformation(trainSentences)
    devSentences = addCharInformation(devSentences)
    testSentences = addCharInformation(testSentences)

    labelSet = set()
    words = {}

    # :: lowercase words dict ::
    for dataset in [trainSentences, devSentences, testSentences]:
        for sentence in dataset:
            for token,char,label in sentence:
                labelSet.add(label.strip())
                words[token.lower()] = True

    # :: Create a mapping for the labels ::
    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    # :: Hard coded case lookup ::
    case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

    # :: Read in word embeddings ::
    word2Idx, wordEmbeddings = get_word_embeddings(glove_embeddings_path, words)

    char2Idx = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)

    train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx))
    dev_set = padding(createMatrices(devSentences,word2Idx, label2Idx, case2Idx,char2Idx))
    test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

    idx2Label = {v: k for k, v in label2Idx.items()}
    np.save(os.path.join(ROOTDIR, "models", "idx2Label.npy"), idx2Label)
    np.save(os.path.join(ROOTDIR, "models", "word2Idx.npy"), word2Idx)

    train_batch, train_batch_len = createBatches(train_set)
    dev_batch, dev_batch_len = createBatches(dev_set)
    test_batch, test_batch_len = createBatches(test_set)

    model = get_model(1, wordEmbeddings, caseEmbeddings, char2Idx, label2Idx)

    model = train_model(
            model,
            epochs,
            train_batch_len,
            train_batch,
            dev_batch_len,
            dev_batch,
            idx2Label,
            label2Idx,
    )

    #   Performance on test dataset
    predLabels, correctLabels = tag_dataset(model, test_batch)
    pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
    print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

    model.save(os.path.join(ROOTDIR, "models", "model.h5"))


if __name__=="__main__":

    main(50, "../embeddings/glove.6B.100d.txt")
