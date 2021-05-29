import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Summary Generator')
top.configure(background='white')
text =""""""

def generateSummary(file_path):
    global label_packed
    f=open(file_path,'r')
    text=f.read()
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    print ("original text preprocessed: \n", preprocess_text)
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    messageVar = Message(top, text = output,font='20')
    messageVar.config(bg='lightgreen',width=500)
    messageVar.pack( pady=35)

   

def show_classify_button(file_path):
    classify_b=Button(top,text="Generate Summary",command=lambda: generateSummary(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_file():
    try:
        file_path=filedialog.askopenfilename()
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload a text file",command=upload_file,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
heading = Label(top, text="Quick Peak",pady=20, font=('arial',20,'bold'))
heading.configure(foreground='#364156')
heading.pack()
top.mainloop()
