import sys
import getopt

arg_window="" 
arg_start=""
arg_end=""
arg_output=""
arg_part=""
arg_sample=""
arg_weight=""

def argpars(argv):

    global arg_window 
    global arg_start
    global arg_end
    global arg_output
    global arg_part
    global arg_sample
    global arg_weight

    arg_help = "{0} -w <window size> -s <startpoint> -e <endpoint>  -o <output> -p <parts numbre> -S <Sample size> -W <weight percentage>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "h:w:s:e:o:p:S:W:", ["help", "window=", 
        "start=", "end=","output=","part=","sample=","weight="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-w", "--window"):
            arg_window = arg
        elif opt in ("-s", "--start"):
            arg_start = arg
        elif opt in ("-e", "--end"):
            arg_end = arg
        elif opt in ("-o", "--output"):
            arg_output = arg
        elif opt in ("-p", "--part"):
            arg_part = arg
        elif opt in ("-W", "--weight"):
            arg_weight = arg        
        elif opt in ("-S", "--sample"):
            arg_sample = arg

    print('window:', arg_window)
    print('start:', arg_start)
    print('end:', arg_end)
    print('output:', arg_output)
    print('part:', arg_part)
    print('weight:', arg_weight)
    print('sample:', arg_sample)


#if __name__ == "__main__":
argpars(sys.argv)