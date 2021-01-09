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



