#!/usr/bin/env python3
# Copyright 2021  Jiayu DU
#           2021  Xiaomi Corporation (Author: Yongqing Wang)

import sys

def asr_text_post_processing(text):
    return text.lower()

if __name__ == '__main__':
    for line in sys.stdin.readlines():
        if line.strip():
            cols = line.strip().split(maxsplit=1)
            key = cols[0]
            text = ''
            if len(cols) == 2:
                text = cols[1]
            print(F'{key} {asr_text_post_processing(text)}')
