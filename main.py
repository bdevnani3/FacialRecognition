from __future__ import print_function
__author__ = 'bhavika'

# Import the required modules
import cv2,os
import argparse
import trainRecognizer
from recognize import recognize

def argumentparser():
    parser = argparse.ArgumentParser(description='Recognize Faces')
    parser.add_argument('-recognize',type=str,action='store',dest = "r_path",
                        default = None,
                       help='Once given path name to image it will recognize the person in the picture')
    parser.add_argument('-train', action='store',dest = "t_path",type=str,default = None,
                        help='Given a path name of a folder, it will use the contents to train a recognizer.')
    parser.add_argument('-trainAll', action='store',dest = "ta_path",type=str,default = None,
                        help='Given a path name of a directory, it will use the contents to train many recognizers.')
    parser.add_argument('-recognizeAll', action='store',dest = "ra_path",type=str,default = None,
                        help='Given a path name of a directory, it will recognize all of the contents')
    parser.add_argument('-c', action='store_true', default=False,
                        dest='boolean_switch',
                        help='Set a switch to true')
    args = parser.parse_args()
    return args

def main():
    args = argumentparser()
    if args.t_path:
        trainRecognizer.train(args.t_path,None)
    elif args.r_path:
        recognize(args.r_path)
    elif args.ta_path:
        paths = [x[0] for x in os.walk(args.ta_path)][1:]
        for i in paths:
            trainRecognizer.train(i)
    elif args.ra_path:
        paths = trainRecognizer.listdir_nohidden(args.ra_path)
        for path in paths:
            print(path)
            recognize(path)
    elif args.store_true:
        pass
    else :
        print("Please enter a command!")

main()

