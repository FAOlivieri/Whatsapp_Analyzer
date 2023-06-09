# Imports de Flask
import shutil

from flask import Flask, render_template, request, jsonify, session
from uuid import uuid4
import os
from werkzeug.utils import secure_filename

# Imports de WppAnalyzer

import pandas as pd
import re
import io
from datetime import datetime
import calendar
from plotnine import *  # NOQA
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family': 'serif',
        'weight': 'bold',
        'size': 10}

mpl.rc('font', **font)
mpl.use("Agg")

plotwidth = 8
plotheight = 4

from os import path
import itertools as it
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from matplotlib.font_manager import FontProperties
import string
# import bokeh as bk
import itertools as it
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from os import path
from matplotlib.font_manager import FontProperties
import string

# import tidypython


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STOPWORD_FOLDER = 'stopwords'
app.config['STOPWORD_FOLDER'] = STOPWORD_FOLDER

STATIC_FOLDER = 'static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

app.secret_key = 'fedefedefede'


@app.route('/')
def index():
    if 'uuid' not in session:
        session['uuid'] = str(uuid4())
    return render_template('index.html')

@app.route('/delete', methods=['POST'])
def delete():
    print("Called to delete")
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], session['uuid'])
    static_path = os.path.join(app.config['STATIC_FOLDER'], session['uuid'])

    shutil.rmtree(folder_path)
    shutil.rmtree(static_path)

    if (os.path.isdir(static_path) or os.path.isdir(folder_path)):
        return jsonify({'error': 'Couldn\'t remove files'})
    else:
        return jsonify({'success': "success"})


@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is selected
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected.'})

    file = request.files['file']

    # Check if the file has a valid extension
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], session['uuid'])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, file.filename)
        file.save(file_path)

        # Process the text file and generate an image (Replace this with your own logic)
        # Assuming the generated image is saved as 'output.jpg'

        image_urls = process_text_file(file_path)

        return jsonify({'image_urls': image_urls})

    return jsonify({'error': 'Invalid file type.'})

def allowed_file(filename):
    allowed_extensions = {'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_text_file(file_path):
    # Add your logic here to process the text file and generate an image
    # This is just a placeholder code that returns a sample image path
    # Replace this code with your own image generation logic

    # For demonstration purposes, copying the uploaded text file as an image
    static_path = os.path.join(app.config['STATIC_FOLDER'], session['uuid'])
    if not os.path.exists(static_path):
        os.makedirs(static_path)

    input_file = io.open(file_path, "rt", encoding='utf-8')
    data = input_file.readlines()
    input_file.close()

    stopword_path = os.path.join(app.config['STOPWORD_FOLDER'])
    stopword_file_path = os.path.join(stopword_path, "stopwords-all.txt")

    stopwordsfile = open(stopword_file_path, "rt", encoding='utf-8')
    stopwords = stopwordsfile.read().splitlines()
    stopwordsfile.close()

    df = parse_data(data)
    df['User'] = df['User'].str.replace('\W', '')
    users = df.User.unique()
    userdict = {i: users[i] for i in range(0, len(users))}

    image_urls=[]

    image_url = messages_per_user(df, users, static_path)
    image_urls.append(image_url)

    image_url = average_message_length(df, users, static_path)
    image_urls.append(image_url)

    image_url = weekday(df, static_path)
    image_urls.append(image_url)

    image_url = time_of_day(df, static_path)
    image_urls.append(image_url)

    image_url = by_date(df, static_path)
    image_urls.append(image_url)

    image_url = interaction(df, users,userdict, static_path)
    image_urls.append(image_url)

    do_wordclouds=False
    if (do_wordclouds):
        image_urls_partial=wordclouds(df,users,stopwords,static_path)
        image_urls=image_urls+image_urls_partial


    print(image_urls)
    return image_urls

def parse_data(data):
    pattern = re.compile("\d{1,2}/\d{1,2}/\d{1,2}.+-.+")

    for index, line in enumerate(data):
        while index < len(data) - 1 and not bool(pattern.match(data[index + 1])):
            data[index] = data[index] + data.pop(index + 1)
    datanot = [x for x in data if not bool(pattern.match(x))]
    datamulti = [x for x in data if x.count("\n") >= 2]
    data = [x for x in data if bool(pattern.match(x))]
    data = [x for x in data if x.count(":") >= 2]

    # Agregar el 0-padding a la fecha y hora
    for index, line in enumerate(data):
        if line[1] == "/":
            data[index] = "0" + line
    for index, line in enumerate(data):
        if line[4] == "/":
            data[index] = line[:3] + "0" + line[3:]

    if (data[0][8] == ","):
        monthfirst = True
        nchck = 11
    else:
        monthfirst = False
        nchck = 10

    for index, line in enumerate(data):
        if line[nchck] == ":":
            data[index] = line[:nchck - 1] + "0" + line[nchck - 1:]

    # Cargar el dataframe
    d = []

    format12h = (data[0][17] == "M")
    end_of_date = min(data[0].find(","), data[0].find(" "))
    has_comma = (data[0][end_of_date] == ",")
    long_year = end_of_date == 10
    if not (long_year):
        for index, line in enumerate(data):
            data[index]=line[:6]+"20"+line[6:]

    print(data[0])

    for index, line in enumerate(data):
        if has_comma:
            if monthfirst:
                if format12h:
                    dtm = datetime.strptime(line[:20], "%m/%d/%Y, %I:%M %p")
                    userend = data[index].find(":", 20)
                    user = line[23:userend]
                else:
                    dtm = datetime.strptime(line[:17], "%m/%d/%Y, %H:%M")
                    userend = data[index].find(":", 17)
                    user = line[20:userend]
            else:
                if format12h:
                    dtm = datetime.strptime(line[:20], "%d/%m/%Y, %I:%M %p")
                    userend = data[index].find(":", 20)
                    user = line[23:userend]
                else:
                    dtm = datetime.strptime(line[:17], "%d/%m/%Y, %H:%M")
                    userend = data[index].find(":", 17)
                    user = line[20:userend]
        else:
            if monthfirst:
                if format12h:
                    dtm = datetime.strptime(line[:19], "%m/%d/%Y %I:%M %p")
                    userend = data[index].find(":", 19)
                    user = line[22:userend]
                else:
                    dtm = datetime.strptime(line[:16], "%m/%d/%Y %H:%M")
                    userend = data[index].find(":", 16)
                    user = line[19:userend]
            else:
                if format12h:
                    dtm = datetime.strptime(line[:19], "%d/%m/%Y %I:%M %p")
                    userend = data[index].find(":", 19)
                    user = line[22:userend]
                else:
                    dtm = datetime.strptime(line[:16], "%d/%m/%Y %H:%M")
                    userend = data[index].find(":", 16)
                    user = line[19:userend]

        message = line[userend + 1:-1]
        d.append(
            {
                "Datetime": dtm,
                "Date": dtm.date(),
                "Hour": dtm.hour,
                "Minute": dtm.minute,
                "Weekday": dtm.weekday(),
                "WeekdayName": list(calendar.day_abbr)[dtm.weekday()],
                "User": user,
                "Message": message
            }
        )

    df = pd.DataFrame(d)
    return df

def messages_per_user(df, users, static_path):
    msgcounts = df[["User", "Message"]].groupby(["User"])["Message"].count()
    msgcounts = msgcounts.reset_index(name='counts')
    # ggplot(counts) + geom_bar(aes(x="Hour",y="counts",fill="User"), size=1)\
    #    + ylab("Mensajes") + coord_polar("y",start=0) # ESTA PIJA NO ANDA EN PLOTNINE
    fig = plt.figure(1, figsize=(plotwidth, plotheight))
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(users))]
    fig, axs = plt.subplots(1, 1, figsize=(plotwidth, plotheight))
    axs.set_title('Number of messages')
    plt.pie(msgcounts.counts, labels=msgcounts.User, autopct='%1.1f%%', colors=colors)
    output_file = os.path.join(static_path, "N_of_messages.jpg")
    plt.savefig(output_file, dpi=1000, bbox_inches="tight")
    plt.close()
    output_url = request.host_url + 'static/' + session['uuid'] + '/' + "N_of_messages.jpg"
    return output_url


def average_message_length(df, users, static_path):
    df["MsgLen"] = df["Message"].apply(len)
    avg = df[["User", "MsgLen"]].groupby("User")["MsgLen"].mean()
    avg.axes
    fig = plt.figure(1, figsize=(plotwidth, plotheight))
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(users))]
    fig, axs = plt.subplots(1, 1, figsize=(plotwidth, plotheight))
    axs.set_title('Average message length')
    plt.bar(avg.axes[0].tolist(), avg.values, color=colors)
    plt.xticks(rotation=45, ha="right")
    output_file = os.path.join(static_path, "msg_length.jpg")

    plt.savefig(output_file, dpi=1000, bbox_inches="tight")
    plt.close()
    output_url = request.host_url + 'static/' + session['uuid'] + '/' + "msg_length.jpg"
    return output_url

def weekday(df,static_path):
    # Mensajes segun dia de la semana
    counts = df[["User","Weekday","Message","WeekdayName"]].groupby(["User","Weekday"])["Message"].count()
    counts = counts.reset_index(name='counts')
    p = ggplot(counts) + geom_line(aes(x="Weekday",y="counts",color="User"), size=1) \
        + scale_x_continuous(breaks=(0,1,2,3,4,5,6), labels=["Mon","Tue", "Wed", "Thu","Fri","Sat","Sun"]) \
        + ylab("Messages") + ggtitle("Messages by weekday")

    p.save("weekday.jpg",format="jpg", path=static_path,dpi=1000,height=plotheight, width=plotwidth,units = 'in')
    output_url = request.host_url + 'static/' + session['uuid'] + '/' + "weekday.jpg"
    return output_url

def by_date(df,static_path):
    counts = df[["User","Date","Message"]].groupby(["Date"])["Message"].count()
    counts = counts.reset_index(name='counts')
    p = ggplot(counts) + geom_line(aes(x="Date",y="counts",group=1), size=1) \
        + ylab("Messages") + scale_x_date(major="months", minor="weeks") + theme(axis_text_x  = element_text(angle = 45, hjust = 1))+ ggtitle("Messages by date")
    p.save("date.jpg",format="jpg", path=static_path,dpi=1000,height=plotheight, width=plotwidth,units = 'in')
    output_url = request.host_url + 'static/' + session['uuid'] + '/' + "date.jpg"
    return output_url

def time_of_day(df,static_path):
# Mensajes segun hora del dia
    counts = df[["User","Hour","Message"]].groupby(["User","Hour"])["Message"].count()
    counts = counts.reset_index(name='counts')
    p = ggplot(counts) + geom_line(aes(x="Hour",y="counts",color="User"), size=1) \
        + ylab("Messages") + ggtitle("Messages by time of day")
    p.save("hour.jpg",format="jpg", path=static_path,dpi=1000,height=plotheight, width=plotwidth,units = 'in')
    output_url = request.host_url + 'static/' + session['uuid'] + '/' + "hour.jpg"
    return output_url

def interaction(df,users,userdict,static_path):
    # Correlación
    df2=df
    df2["prevUser"] = df2["User"].shift(1)
    df2["prevUser"][0] = df2["User"][0]
    mat = np.zeros((len(users),len(users)))
    for row in df2.iterrows():
        senderindex=np.where(users==row[1]["User"])[0][0]
        previndex = np.where(users == row[1]["prevUser"])[0][0]
        if senderindex!=previndex:
            mat[senderindex,previndex] += 1

    mat
    #g = nx.from_numpy_matrix(np.matrix(mat), create_using=nx.MultiDiGraph) # Para hacerlo bidireccional
    g = nx.from_numpy_array(np.matrix(mat), create_using=nx.Graph)
    pos = nx.circular_layout(g)
    edges = g.edges()

    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
    edges
    weights
    weightsreduced=[]
    for item in weights:
        weightsreduced.append(item/250)
    weightsreduced


    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])

    fig = plt.figure(figsize=(plotwidth,plotheight))
    f, axs = plt.subplots(1,1,figsize=(plotwidth,plotheight))
    nx.draw(g, pos, labels=userdict, edge_color=weightsreduced, edge_cmap=cmap, width=weightsreduced)

    plt.tight_layout()
    plt.autoscale()

    output_file = os.path.join(static_path, "interaction.jpg")

    plt.savefig(output_file, dpi=1000, bbox_inches="tight")
    plt.close()
    output_url = request.host_url + 'static/' + session['uuid'] + '/' + "interaction.jpg"
    return output_url

def wordclouds(df,users,stopwords,static_path):
    palabras=[]
    subdf=df[["Message"]]
    fulllist = list(it.chain.from_iterable(map(str.split, (subdf["Message"]))))
    fulllist = [x.lower() for x in fulllist]
    fulllist = [x.replace("\"","") for x in fulllist]
    fulllist = [x.replace("(","") for x in fulllist]
    fulllist = [x.replace(")","") for x in fulllist]
    fulllist = [x.replace(",","") for x in fulllist]
    fulllist
    stopwords.append("jajajaja")
    stopwords.append("jaja")
    stopwords.append("jajajajaja")
    stopwords.append("<media")
    stopwords.append("omitted>")
    stopwords.append("<multimedia")
    stopwords.append("omitido>")
    stopwords
    grouplist = [x for x in fulllist if x not in stopwords]
    grouplist
    counts= Counter(grouplist)
    common=counts.most_common(50)
    common

    ignorechars = "\\`{}[]()>!\",🏻🏼🏽🏾🏿"

    for user in users:
        print (user)
        subdf=df[["Message"]].loc[df["User"]==user]
        listofwords= list(it.chain.from_iterable(map(str.split, (subdf["Message"]))))
        listofwords = [x.lower() for x in listofwords]

        for c in ignorechars:
            listofwords = [x.replace(c,"") for x in listofwords]


        listofwords = [x for x in listofwords if x not in stopwords]    #Esta linea es una prueba
        counts = Counter(listofwords)
        common=counts.most_common(10)
        palabras.append([user,listofwords]) #usar la lista completa de palabras


    palabrasdif=[]
    threshold=500

    for item in palabras:
        newwordlist=list(item[1])
        for item2 in palabras:
            if item2!=item:
                commonwords=list(list(zip(*Counter(item2[1]).most_common(threshold)))[0])
                commonwords
                newwordlist = np.array([el for el in newwordlist if el not in commonwords])
        palabrasdif.append([item[0],newwordlist])

    palabrasdif
    palabrasdif.append(["Group", grouplist])
    users2=np.append(users, "Group")
    users
    Counter(palabrasdif[0][1]).most_common(10)
    normal_word = r"(?:\w[\w']+)"
    ascii_art = r"(?:[{punctuation}][{punctuation}]+)".format(punctuation=string.punctuation)
    # a single character that is not alpha_numeric or other ascii printable
    emoji = r"(?:[^\s])(?<![\w{ascii_printable}])".format(ascii_printable=string.printable)
    regexp = r"{normal_word}|{ascii_art}|{emoji}".format(normal_word=normal_word, ascii_art=ascii_art,
                                                         emoji=emoji)
    font_path = path.join("C:\\Windows", 'fonts', 'Symbola', 'Symbola.ttf')
    wordcloud = WordCloud(font_path="C:/Windows/Fonts/seguisym.ttf",width=1000,height=1000,max_words=50,min_font_size=10,stopwords=stopwords,background_color="white",regexp=regexp)
    #wordcloud = WordCloud(font_path=font_path,width=1000,height=1000,max_words=50,min_font_size=10,stopwords=stopwords,background_color="white",regexp=regexp)

    output_urls=[]

    for i, user in enumerate(users2):
        wordcloud_gen= wordcloud.generate_from_frequencies(Counter(palabrasdif[i][1]))
        # plot the WordCloud image
        fig = plt.figure(figsize=(6,6))
        f, axs = plt.subplots(1,1,figsize=(6,6))
        plt.imshow(wordcloud_gen, interpolation='bilinear')
        plt.axis("off")
        plt.title(user,fontdict={"fontsize":25}, pad=20)
        plt.tight_layout(pad = 0)
        output_file = os.path.join(static_path, (user+".jpg"))

        plt.savefig(output_file, dpi=1000, bbox_inches="tight")
        plt.close()
        output_url = request.host_url + 'static/' + session['uuid'] + '/' + (user+".jpg")
        output_urls.append(output_url)
    return output_urls

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=80, threaded=True)
