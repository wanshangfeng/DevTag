# DevTag

___Bacground___: The recent rise in embedded system technologies has insti-
gated a significant increase in the development and deployment of Internet-of-Things (IoT) devices, including including routers, webcams, and
network printers, causing security concerns.

___DevTag___ is a tool that recognizes information about IoT devices, including a ___rule-based approach___ and a ___model-based approach___.

The input is the remote host's banner in the application-layer protocol, and the output is the tag of the remote host. 
The Tag format is the <device_type, vendor, product_info>.


## The Rule-based Approach 

As far, we use three popular sources (listed by Table 1) to generate rules of IoT devices, including [NMAP](https://nmap.org/), [ZTAG](https://github.com/zmap/ztag), and [ARE](https://www.usenix.org/conference/usenixsecurity18/presentation/feng). 

Source    | Original Format            |   How the rules are stored |  Protocol|
--------- | --------                   |--------        | ---------|
NMAP      | Regex-> Device Tag        |    File      |  FTP, HTTP, RTSP, Telnet|
ZTAG      | String/Regex-> Device Tag |   Script    | FTP, HTTP, Telnet|
ARE       | String -> Device Tag       |  File    |  FTP, HTTP, RTSP, Telnet|

Note that those rules use different formats and name conventions for IoT devices. 
To integrate consistent rules, we revise name conventions for all IoT rules and
use a unified format to represent them. 
```
<String/Regex>  -> <device_type, vendor, product_info>
```

If a banner of host is matched with <String/Regex> of rule, DevTag provides a tag to this host.


### The rule-based approach: Usage
```python
python __main__.py -p <protocol> -f <filename> -T <all/part> -dType <device type> -ven <vendor name>
```

| Parameter         | Help           |
|---------  | --------                 |
|protocol |  FTP, HTTP, RTSP, Telnet|
|filename | JSON file (banners of hosts) |
|all/part | indicates what rules to use   |  
|device type | uses rules belong to this *type* |
|vendor name | uses rules belong to this *vendor* |


## The Model-based Approach 

### Requirements
The tool is implemented in Python 3. To install needed packages use:
```
pip3 install -r requirements.txt
```

### Model
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
