#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from twitter_text_preprocess import sanitize


#Generate tokenized corpus from annotated tweets file
def raw_tweet2tokens(input_annotated_file, output_tokenized_corpus_file, tweets_column_name):
    original_frame = pd.read_csv(input_annotated_file)
    tweets_column_list = list(original_frame[tweets_column_name])
    tweets_str = ''.join(tweets_column_list[0:len(tweets_column_list)])
    tweets_tokenized_str = sanitize(tweets_str)
    with open(output_tokenized_corpus_file, 'a+') as f:
        f.write(tweets_tokenized_str)

#Do tokenization for tweets in annotated file before transfering tokens to vectors
def annotatedfile_tokenization(input_annotated_file, output_tokenized_file, tweets_column_name):
    original_frame = pd.read_csv(input_annotated_file)
    for column in range(len(original_frame)):
        original_frame[tweets_column_name].iloc[column] = sanitize(original_frame[tweets_column_name].iloc[column])
    original_frame.to_csv(output_tokenized_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help = "define the location of the input;", required = True)
    parser.add_argument(
        '-ot', '--output_txt', help = "define the location of the output;", required = True)
    parser.add_argument(
        '-oc', '--output_csv', help = "define the location of the output;", required = True)
    parser.add_argument(
        '-co' '--column', help = "point out the name of the column containing tweets;", dest = 'column',required = True)

    args = parser.parse_args()


    raw_tweet2tokens(args.input, args.output_txt, args.column)
    annotatedfile_tokenization(args.input, args.output_csv, args.column)
