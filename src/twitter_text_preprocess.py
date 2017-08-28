#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
preprocess-twitter.py (https://gist.github.com/tokestermw/cb87a97113da12acb388)
python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"
Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import re
from ftfy import fix_text
import string
import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')



FLAGS = re.MULTILINE | re.DOTALL
nonvalid_characters_p = re.compile("[^a-zA-Z0-9#\*\-_\s]")

   
def re_sub(pattern, repl, text):
    return re.sub(pattern, repl, text, flags=FLAGS)

# def hashtag(text):
#     text = text.group()
#     hashtag_body = text[1:]
#     if hashtag_body.isupper():
#         result = "<hashtag> {} <allcaps>".format(hashtag_body)
#     else:
#         result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
#     return result

def hashtag(text):
    text = text.group()
    
    return text[1:] + " <hashtag> "


def hashtag_converter(text):
    return re_sub(r"#\S+", hashtag, text)

#restore abbr, eg. I've --> I have
def abbr_restore(text):
    #logger.info("restoring abbr")
    text = re_sub("i\'ve", "i have", text)
#    text = re_sub("don\'t", "do not", text)
    text = re_sub("n\'t", " not", text)
    text = re_sub("i\'d like", "i would like", text)
    text = re_sub("i\'d (?!like)", "i had ", text)
    text = re_sub("that\'s", "that is", text)
    text = re_sub("it\'s", "it is", text)    
    return text


def alpha_and_number_only(text):
    text = re.sub(nonvalid_characters_p, ' ', text)
    return text

def pop_words_transformation(text):
    #logger.info("transforming pop self-defined words")
    text = re_sub(r'ha[ha]+', '<symbollaugh>', text)
#    text = re_sub(r'Ha[ha]+', 'symbollaugh', text)
    text = re_sub(r'hua[hua]+', '<symbollaugh>', text)
#    text = re_sub(r'Hua[hua]+', 'symbollaugh', text)
    text = re_sub(r'ja[ja]+', '<symbollaugh>', text)
    text = re_sub(r'ja[ja]+j', '<symbollaugh>',text)
    text = re_sub(r'kno[o]+w', 'know', text)
    text = re_sub(r'goo[o]+d', 'good', text)
    text = re_sub(r'tir[r]+e[e]+d', 'tired', text)
    return text

def remove_punctuation(text):
    #logger.info("removing punctuation")
    #return ' '.join(word.strip(string.punctuation) for word in text.split())
    text = re_sub(r'(\.|\?|;|!|,|~)(\s+|$)', ' ', text)

    return text 

def url_converter(text):
    #logger.info("processing <url>")
    #text = re_sub(r"htt\S+", " <url> ", text)
    #text = re_sub("htt", "", text)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>", text)
    return text

def user_converter(text):
    #logger.info("processing <username>")
    return re_sub(r"@\w+", " <user> ", text)

def emoji_converter(text):
    #logger.info("processing <emoji>")
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ", text)
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ", text)
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ", text)
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ", text)
    text = re_sub(r"<3"," <heart> ", text)
    return text    

def num_converter(text):
    #logger.info("processing <number>")
    return re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"

def allcaps_converter(text):
    #logger.info("processing <allcaps>")
    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    return re_sub(r"([A-Z]){2,}", allcaps, text)


def special_repeat_converter(text):
    #logger.info("processing <repeat> and <elon>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat> ", text)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ", text)
    return text

non_ascii_p = re.compile(r'[^\x00-\x7F]+')

def clean_text(text):
    text = fix_text(text.replace('\r\n',' ').replace('\n',' ').replace('\r',' '))

    text = re.sub(non_ascii_p, '', text)

    return text.strip()

def sanitize(text):
    #logger.info("sanitizing tweet")

    text = clean_text(text)
    text = url_converter(text)
    text = re_sub(r"/"," / ", text)    
    text = user_converter(text)
    #text = emoji_converter(text)
    #text = num_converter(text)
    text = hashtag_converter(text)
    #text = special_repeat_converter(text)
    
    #text = abbr_restore(text)
    #text = alpha_and_number_only(text)
    
    #text = pop_words_transformation(text)
    text = allcaps_converter(text)
    text = remove_punctuation(text)
    return text.strip()

def sanitize_nofunccall(text):

    text = clean_text(text)
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>", text, flags=FLAGS)
    text = re.sub(r"/"," / ", text, flags=FLAGS)
    text = re.sub(r"@\w+", " <user> ", text, flags=FLAGS)
    text = re.sub(r"#(\S+)", hashtag, text, flags=FLAGS)
    text = re.sub(r"([A-Z]){2,}", allcaps, text, flags=FLAGS)
    text = re.sub(r'(\.|\?|;|!|,|~)(\s+|$)', ' ', text, flags=FLAGS)
    return text.strip()

if __name__ == '__main__':

    text = "I TEST alllll kinds of #whatever and #HASHTAGS, @mentions 300,000 1.5 and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    logger.info(sanitize(text))
    logger.info(sanitize_nofunccall(text))
    # _, text = sys.argv
    # if text == "test":
    #     text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    # tokens = sanitize(text)
    # print(tokens)
