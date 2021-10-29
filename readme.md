# Description
There has been an information explosion in the world. Using **Knowledge** Graphs can extract information effectively. 
Our ultimate goal is to build a Knowledge Graphs system, which's input is an article and the output is Knowledge Graph. 
In order to complete this system, we need to finish two main tasks, which are **Named Entity Recognition(NER)** and **Relation Extraction(RE)**.
Our initial thought is that passing an article to the NER model so that we can obtain Entity Objects. 
Then pass these Entity Objects to the RE model to understand the relationship between these Entity Objects. 
After that we can organize the Knowledge Graph from an article. 
In the project, we attempt to train a Named Entity Recognition model using pre-trained **BERT**.

# Dataset
**WikiANN** is a multilingual named entity recognition dataset consisting of Wikipedia articles annotated with **LOC** (location), **PER** (person), and **ORG** (organisation) tags in the IOB2 format. 
The language tag we choose is **en**(English) with the size of train, validation and test dataset are 20000, 10000 and 10000 separately. 

# Preprocess
BERT uses the WordPiece tokenizer so that some words be tokenized will split into more than one tokens. Therefore, the length of labels may not be equal to the length of tokenized sentence. 
The function `less_than_max_len` is designed to determine the index **K** of the last token in a sentence such that the length of tokenized top-**K** token is smaller than the length of `max_len`.

Besides, the label need to be propagated by the length of tokenized token.

E.g.
```python3
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = ['R.H.','Saunders','(','St.','Lawrence','River',')','(','968','MW',')']
labels = [3, 4, 0, 3, 4, 4, 0, 0, 0, 0, 0]
encoded_tokens = tokenizer(tokens, add_special_tokens=False)

print(encoded_tokens['input_ids'])
>>> [[1054, 1012, 1044, 1012], [15247], [1006], [2358, 1012], [5623], [2314], [1007], [1006], [5986, 2620], [12464], [1007]]

num_of_tokens = [len(x) for x in encoded_tokens['input_ids']]
print(num_of_tokens)
>>> [4, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1]

propagate_labels = []
for i in range(len(num_of_tokens)):
    propagate_labels += [labels[i]]* num_of_tokens[i]
    
print(propagate_labels)
>>> [3, 3, 3, 3, 4, 0, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0]
```
# Model
We use **BertForTokenClassification** model, which inherits from PreTrainedModel with **bert-base-uncased**.



# Evaluation
We know that

<img src="https://latex.codecogs.com/svg.image?\textbf{F1&space;score}&space;=&space;\frac{2\times\textbf{Precision}\times\textbf{Recall}}{\textbf{Precision}&plus;\textbf{Recall}}\&space;\&space;with" title="\textbf{F1 score} = \frac{2\times\textbf{Precision}\times\textbf{Recall}}{\textbf{Precision}+\textbf{Recall}}\ \ with" />

<img src="https://latex.codecogs.com/svg.image?\textbf{Precision}&space;=&space;\frac{\textbf{TP}}{\textbf{TP}&plus;\textbf{FP}}\&space;\&space;and\&space;\&space;\textbf{Recall}&space;=&space;\frac{\textbf{TP}}{\textbf{TP}&plus;\textbf{FN}}" title="\textbf{Precision} = \frac{\textbf{TP}}{\textbf{TP}+\textbf{FP}}\ \ and\ \ \textbf{Recall} = \frac{\textbf{TP}}{\textbf{TP}+\textbf{FN}}" />

Because we evaluate model at **entity-level**, <img src="https://latex.codecogs.com/svg.image?\left&space;(&space;{\textbf{TP}&plus;\textbf{FP}}&space;\right&space;)" title="\left ( {\textbf{TP}+\textbf{FP}} \right )" /> means that the total number of **Named Entities in predictive label** and <img src="https://latex.codecogs.com/svg.image?\left&space;(&space;{\textbf{TP}&plus;\textbf{FN}}&space;\right&space;)" title="\left ( {\textbf{TP}+\textbf{FN}} \right )" /> means that the total number of **Named Entities in true label** and <img src="https://latex.codecogs.com/svg.image?\textbf{TP}" title="\textbf{TP}" /> means that the number of cases, which's Named Entity in predictive label is equal to the one in true label.

E.g.
    
    Case1
    
    true = [['O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'B-ORG', 'O', 'B-LOC']]
    pred = [['O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'B-ORG', 'O',   'O'  ]]
    
    Then, Precision = 2/2 = 1 and Recall = 2/3 , so F1 score = 4/5
    
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    Case2
    
    true = [['O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-PER'], ['O', 'B-ORG', 'O',   'O'  ]]
    pred = [['O', 'B-LOC', 'I-LOC',   'O',   'O', 'B-PER'], ['O', 'B-ORG', 'O', 'B-PER']]

    Then, Precision = 2/4 = 1/2 and Recall = 2/3, so F1 score = 4/7

# Result
We use AdamW as optimizer and assign epochs to 10. In every epoch, we record F1-score on train, validation and test dataset separately. Saving model if F1-score is greater than previous epoch.
The best F1-score in test dataset is close to 80%. In the future, we can try another pre-trained tokenizer and model, and using Learning Rate Schedules to adjust learning rate. These ways may improve F1-score of model.

| Epoch | Training | Validation |  Test  |
| :---: | :------: | :--------: |  :---: |
|   1   |  76.01   |    76.97   |  76.35 |
|   2   |  80.84   |    76.02   |  75.64 |
|   3   |  79.94   |    76.73   |  76.31 |
|   4   |  86.26   |    78.39   |  77.80 |
|_**5**_|  89.53   |_**79.25**_ |_**78.40**_|
|   6   |  90.06   |    76.85   |  75.86 |
|   7   |  90.84   |    77.59   |  77.02 |
|   8   |  91.85   |    78.59   |  78.27 |
|   9   |  93.07   |    76.24   |  75.81 |
|  10   |  94.47   |    77.54   |  77.02 |
    


# References
1. [An Introduction to Knowledge Graphs](https://ai.stanford.edu/blog/introduction-to-knowledge-graphs/)
2. [Wikiann Dataset](https://huggingface.co/datasets/wikiann)
3. [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer)
4. [BertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification)
5. [seqeval](https://pypi.org/project/seqeval/)


