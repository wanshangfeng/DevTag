import argparse
import sys
import json
import os

from decoders import JSONDecoder
from encoders import JSONEncoder
from log import Logger
from banner_preprocess import clean_banner
from recognition import tag_banner
import path_config

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
    parser.add_argument('-f', '--filename', type=argparse.FileType('r'))
    parser.add_argument('-D', '--decoder', type=JSONDecoder)
    parser.add_argument('-E', '--encoder', type=JSONEncoder)
    parser.add_argument('-T', '--tag', default="part", type=str, choices=["all", "part"])
    parser.add_argument('-dType', '--device_type', default="all", type=str)
    parser.add_argument('-ven', '--vendor', default="all", type=str)

    args = parser.parse_args()

    logger = Logger(args.logfile, args.loglevel)

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

    dirname, fname = os.path.split(os.path.abspath(__file__))
    file = dirname + '/'+path_config.test_path+ '/DevTag-'+protocol+'.json'

    if part_or_all == 'part':
        logger.info("The DevTag provides the part tag")
    if part_or_all == "all":
        logger.info("The DevTag provides the complete tags")
    
    start_time = datetime.utcnow()

    for line in args.filename:
        line_to_json = json.loads(line)
        banner = line_to_json['banner']
        tag_list = tag_banner(protocol, banner, device_type, vendor)
        
        if tag_list is None:
            logger.info("don't find a tag")
            
            with open(file, 'a', encoding='utf-8') as f:
                f.write('{"application": "unknown", ' +
                        '"device_type": "unknown", ' +
                        '"product": "unknown", '+
                        '"vendor": "unknown"} '+ '\n')
                f.write('==================================')
                f.write("\n")

            continue

        #delete same tag
        tag_delete_same_list = []
        [tag_delete_same_list.append(i) for i in tag_list if not i in tag_delete_same_list]
            
        if part_or_all == "part":
            first_tag = tag_delete_same_list[0]
            with open(file, 'a', encoding='utf-8') as f:
                json.dump(first_tag, f, sort_keys=True)
                f.write('\n')
        if part_or_all == "all":
            for tag in tag_delete_same_list:
                with open(file, 'a', encoding='utf-8') as f:
                    json.dump(tag, f, sort_keys=True)
                    f.write('\n')
            with open(file, 'a', encoding='utf-8') as f:
                f.write('==================================')
                f.write('\n')    
    end_time = datetime.utcnow()

    duration = end_time - start_time
    logger.info("the running time is %s" % duration.total_seconds())


if __name__ == "__main__":
    main()
