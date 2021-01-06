import argparse
import sys
import json
import os

from decoders import JSONDecoder
from encoders import JSONEncoder
from log import Logger
from preprocess import clean_all
from recongnition import tag_rules

from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--protocol', type=str,
                        choices=["FTP","ftp",'HTTP',"http",'RTSP',"rtsp",'SMTP',"smtp",'TELNET',"telnet"],
                        help="protocol type")
    parser.add_argument('-l', '--logfile', type=argparse.FileType('w'),
                        default=sys.stderr)
    parser.add_argument('-v', '--loglevel', type=int, default=Logger.INFO,
                        choices=range(0, Logger.TRACE + 1))

    #读取文件内容
    parser.add_argument('-f','--filename',type=argparse.FileType('r'))

    #读取json数据内容
    parser.add_argument('-D','--decoder',type=JSONDecoder)
    parser.add_argument('-E','--encoder',type=JSONEncoder)

    #选择tag选择所有的还是部分的
    parser.add_argument('-T','--tag',default="part",type=str,choices=["all","part"])

    parser.add_argument('-dType','--device_type',default="all",type=str)
    parser.add_argument('-ven','--vendor',default="all",type=str)


    args = parser.parse_args()

    logger = Logger(args.logfile,args.loglevel)

    if not args.protocol:
        logger.info("Error: protocol (-P/--protocol) required\n")
        sys.exit(1)

    if not args.filename:
        logger.info("Error: file (-f/--filename) required\n")
        sys.exit(1)
        
    protocol = args.protocol
    part_or_all = args.tag

    device_type = args.device_type
    vendor = args.vendor

    logger.info("device_type: %s" % device_type)
    logger.info("vendor: %s" %vendor)  

    ## 将banner和tag写入文件中
    dirname,filename = os.path.split(os.path.abspath(__file__))
    file = dirname +'/' +'DevTag2.json'

    if part_or_all == 'part':
        logger.info("We will give you the first tag")
    if part_or_all == "all":
        logger.info("We will give you all tags")
    
    start_time = datetime.utcnow()

    for line in args.filename:
        lines = json.loads(line)
        banner = lines['portInfo']['bannerList'][-1]['banner']
        banner = clean_all(protocol,line)
        tag_list = tag_rules(protocol,banner,device_type,vendor)

        if tag_list is None:
            logger.info("don't find a tag")
            
            with open(file,'a',encoding='utf-8') as f:
                f.write("\n")
                f.write('==================================')
                f.write("\n")

            continue
            

        if part_or_all == "part":
            first_tag = tag_list[0]
            with open(file,'a',encoding='utf-8') as f:
                json.dump(first_tag,f,sort_keys=True)
                f.write('\n')
        if part_or_all == "all":
            for tag in tag_list:
                with open(file,'a',encoding='utf-8') as f:
                    json.dump(tag,f,sort_keys=True)
                    f.write('\n')
            with open(file,'a',encoding='utf-8') as f:
                f.write('==================================')
                f.write('\n')    
    end_time = datetime.utcnow()

    duration = end_time - start_time
    logger.info("time has spent %s" % duration.total_seconds())

    
    


if __name__ == "__main__":
    main()
