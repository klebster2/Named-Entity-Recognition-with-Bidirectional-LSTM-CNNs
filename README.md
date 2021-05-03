# Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs
 A keras implementation of Bidirectional-LSTM_CNNs for Named-Entity-Recognition. The original paper can be found at https://arxiv.org/abs/1511.08308


The implementation differs from the original paper in the following ways :
  1) lexicons are not considered
  2) Bucketing is used to speed up the training
  3) nadam optimizer used instead of SGD

You may need to use dos2unix if you are using a unix machine.

# Result 

# Dataset
##  conll-2003 
 # Network Model in paper

 
 # Network Model Constructed Using Keras
 ![alt text](https://raw.githubusercontent.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/master/model.png)
 
 # To run the script
 ```
    python3 nn.py
 ```
 # Requirements
    0) nltk
    1) numpy 
    2) Keras==2.1.2
    3) Tensorflow==1.4.1


## Inference on trained model

```python
from ner import Parser

p = Parser()

p.load_models("models/")

p.predict("Steve Went to Paris")
##Output [('Steve', 'B-PER'), ('went', 'O'), ('to', 'O'), ('Paris', 'B-LOC')]
```
 
 
 
 
