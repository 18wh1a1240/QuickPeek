import pandas as pd
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from pickle import dump, load
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

top=tk.Tk()
top.geometry('800x600')
top.title('Summary Generator')
top.configure(background='black')
text =""""""


def generateSummary(file_path):
    f=open(file_path,'r',encoding='iso-8859-1')
    reviews=pd.read_csv(f)
    print(reviews.shape)
    print(reviews.head())
    print(reviews.isnull().sum())
    reviews = reviews.reset_index(drop=True)
    print(reviews.shape)
    print(reviews.head())
    for i in range(5):
        print("Review #",i+1)
        print(reviews.headlines[i])
        print(reviews.text[i])
        print()
    contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
    def clean_text(text, remove_stopwords = True):
        '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
        text = text.lower()
    
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in contractions:
                    new_text.append(contractions[word])
                else:
                    new_text.append(word)
            text = " ".join(new_text)
    
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
    
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text
    clean_summaries = []
    for summary in reviews.headlines:
        clean_summaries.append(clean_text(summary, remove_stopwords=False))
    print("Summaries are complete.")

    clean_texts = []
    for text in reviews.text:
        clean_texts.append(clean_text(text))
    print("Texts are complete.")
    for i in range(5):
        print("Clean Review #",i+1)
        print(clean_summaries[i])
        print(clean_texts[i])
        print()
    stories = list()
    for i, text in enumerate(clean_texts):
        stories.append({'story': text, 'highlights': clean_summaries[i]})

    dump(stories, open('review_dataset.pkl', 'wb'))
    stories = load(open('review_dataset.pkl', 'rb'))
    print('Loaded Stories %d' % len(stories))
    print(type(stories))
    def count_words(count_dict, text):
        '''Count the number of occurrences of each word in a set of text'''
        for sentence in text:
            for word in sentence.split():
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
    word_counts = {}

    count_words(word_counts, clean_summaries)
    count_words(word_counts, clean_texts)
            
    print("Size of Vocabulary:", len(word_counts))
    embeddings_index = []

    missing_words = 0
    threshold = 20

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
            
    missing_ratio = round(missing_words/len(word_counts),4)*100
            
    print("Number of words missing from CN:", missing_words)
    print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))
    vocab_to_int = {} 

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

# Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    print("Total number of unique words:", len(word_counts))
    print("Number of words we will use:", len(vocab_to_int))
    print("Percent of words we will use: {}%".format(usage_ratio))
    import numpy as np

    embedding_dim = 300
    nb_words = len(vocab_to_int)

    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in vocab_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_embedding_matrix[i] = new_embedding

    print(len(word_embedding_matrix))
    def convert_to_ints(text, word_count, unk_count, eos=False):
        '''Convert words in text to an integer.
        If word is not in vocab_to_int, use UNK's integer.
        Total the number of words and UNKs.
        Add EOS token to the end of texts'''
        ints = []
        for sentence in text:
            sentence_ints = []
            for word in sentence.split():
                word_count += 1
                if word in vocab_to_int:
                    sentence_ints.append(vocab_to_int[word])
                else:
                    sentence_ints.append(vocab_to_int["<UNK>"])
                    unk_count += 1
            if eos:
                sentence_ints.append(vocab_to_int["<EOS>"])
            ints.append(sentence_ints)
        return ints, word_count, unk_count
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

    unk_percent = round(unk_count/word_count,4)*100

    print("Total number of words in headlines:", word_count)
    print("Total number of UNKs in headlines:", unk_count)
    print("Percent of words that are UNK: {}%".format(unk_percent))
    def create_lengths(text):
        '''Create a data frame of the sentence lengths from a text'''
        lengths = []
        for sentence in text:
            lengths.append(len(sentence))
        return pd.DataFrame(lengths, columns=['counts'])
    lengths_summaries = create_lengths(int_summaries)
    lengths_texts = create_lengths(int_texts)

    print("Summaries:")
    print(lengths_summaries.describe())
    print()
    print("Texts:")
    print(lengths_texts.describe())
    print(np.percentile(lengths_texts.counts, 90))
    print(np.percentile(lengths_texts.counts, 95))
    print(np.percentile(lengths_texts.counts, 99))
    print(np.percentile(lengths_summaries.counts, 90))
    print(np.percentile(lengths_summaries.counts, 95))
    print(np.percentile(lengths_summaries.counts, 99))
    def unk_counter(sentence):
        '''Counts the number of time UNK appears in a sentence.'''
        unk_count = 0
        for word in sentence:
            if word == vocab_to_int["<UNK>"]:
                unk_count += 1
        return unk_count
    sorted_summaries = []
    sorted_texts = []
    max_text_length = 84
    max_summary_length = 13
    min_length = 2
    unk_text_limit = 100 
    unk_summary_limit = 100 

    for length in range(min(lengths_texts.counts), max_text_length): 
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
            ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])
        
# Compare lengths to ensure they match
    print(len(sorted_summaries))
    print(len(sorted_texts))
    for i in range(20):
        print("Review #", i + 1)
        print(clean_texts[i])
        print()
    
    messageVar = Message(top, text = clean_summaries[0])
    messageVar.config(bg='lightgreen',width=400)
    messageVar.pack( pady=35)

def show_classify_button(file_path):
    classify_b=Button(top,text="Generate Summary",command=lambda: generateSummary(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='black',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_file():
    try:
        file_path=filedialog.askopenfilename()
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload a text file",command=upload_file,padx=10,pady=5)
upload.configure(background='#364156', foreground='black',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
heading = Label(top, text="Quick Peak",pady=20, font=('arial',20,'bold'))
heading.configure(foreground='#364156')
heading.pack()
top.mainloop()
