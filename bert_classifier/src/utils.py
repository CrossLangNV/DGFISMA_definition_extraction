import sys
import csv

# _csv.Error: field larger than field limit (131072)
# https://stackoverflow.com/a/15063941/5983691

def csv_field_limit():
    maxInt = sys.maxsize
    decrement = True

    while decrement:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True