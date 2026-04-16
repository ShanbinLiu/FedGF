import sys

def print_log(string):
    print(string)
    sys.stdout.flush()
    f = open('output.txt', 'a')
    f.write(string + "\n")
    f.close()