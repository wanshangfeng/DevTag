# DevTag
DevTag is a tool that can recognizes information about IoT devices. Input the corresponding protocol packet data, it will output its corresponding device information including device type, vendor, product name and so on.
<br>
We provide two ways to identify device information, one is rule-based and the other is model-based.


## Requirements
The tool is implemented in Python 3. To install needed packages use:
```
pip3 install -r requirements.txt
```


## Usage
### Based on Rules
```
python __main__.py -p http -f http_25.json -T all -dType camera -ven tp-link
```
### Based on Model
This part provides the following models: 
```
TextCNN, TextRNN, TextRCNN, TextRNN_Att, DPCNN.
```
#### Training
If retraining on the original data set in this project, you only need to execute:
```
python run.py --type train --model <model name> --embedding <random/ pre_trained>
```
If the data set is updated, you need to perform the following steps: 
first extract the pre-trained word vector, and then select the model for training.
```
python utils.py
```
```
python run.py --type train --model <model name> --embedding <random/ pre_trained>
```
In addition, we provide a method for training word vectors based on ```train.txt```:
```
python get_wordvector.py
```
#### Test
```
python run.py --type test --model <model name> --file <path name>
```
The path name is the file path of your test data.
