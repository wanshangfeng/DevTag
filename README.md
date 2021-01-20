# DevTag
DevTag is a tool that can recognize information about IoT devices. 

The input is the remote host's banner in the application-layer protocol, and the output is the tag of the remote host. 

The Tag format is the <device_type, vendor, product_info>.

<br>
We provide two ways to identify device information, one is rule-based and the other is model-based.


## Requirements
The tool is implemented in Python 3. To install needed packages use:
```
pip3 install -r requirements.txt
```


## Rules Introduction
The rules are from [ARE](https://www.usenix.org/conference/usenixsecurity18/presentation/feng), ZTAG and NMAP.
One rule contains a filterword and a tag. The filterword is like a filter. As long as the banner matches the filterword, it will return the tag.


## Usage
### Based on Rules
```
python __main__.py -p <protocol> -f <filename> -T <all/part> -dType <device type> -ven <vendor name>
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
