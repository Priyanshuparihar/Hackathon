import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from pickle import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from googletrans import Translator


########################################## Side bar of the webapp ##########################################
def load_sidebar():
	st.sidebar.subheader("Sentiment Analysis")

	st.sidebar.success("Analyse Positive and Negative Reviews on twitter ")

	st.sidebar.info("The dataset contains both Negative and Positive tweets and we train this dataset\
		and make Predictions to get perfact analysis of the tweets.")

	
	# Radio button for choices in sidebar
	choice=st.sidebar.radio("Enter your choice: ",('Data Description','Exploratory Data Analysis','Model Evaluation','Predictions',"GUIDE"))
	return choice

###########################################################HELP##############################################
def help_bar():
	ch=st.selectbox("select your choice",('Data Description','Exploratory Data Analysis','Model Evaluation','Predictions'))
	if ch=='Data Description':
		st.subheader("Data Description")
		st.info('''
			It includes following points:-\n
			1) Shows the dataset which has having choice to select top n-number of rows and bottom n-number of rows\n
			2) Shows how many number of rows and columns of  the datasets \n
			3) It display types of tweets i.e whether tweet is negative or positive\n
			4) It display table of description of dataset columns i.e shows columns satistical observations, columns unique values ,their count etc\n''')

	elif ch=='Exploratory Data Analysis':
		st.subheader("Data Visualisation")
		st.success('''
			In this we analyse dataset via plot different types of charts .\n
			First of all '0' represents negative tweets and '1' represents positive tweets.\n
			1) Barchart:- With this we analyse there are equal amount of negative and positive tweets\n
			2) Histogram:- Through this we can say that neagtive tweets length has more than 800 characters
			positive tweets having less than 800 characters.\n
			3) Wordcloud:- In this we analyse frequency/importance of each word in both positive/negative tweets.''')

	elif ch=='Model Evaluation':
		st.subheader("Model Evaluation")
		st.info(''' 
			In this we evaluate our trained model.\n
			It includes following points :-\n
			1) Confusion Matrix :- It shows how efficiently our model is working. \n''')
		st.image("image/confusion-matrix.png",use_column_width=True)
		st.info('''
			2) Precision :- What percentage of Positive precision made were correct.\n
			\tMathematical Formula = True positive/(True positive + False positive)\n
			3) Recall :- What percentage of actual positive values were correctly classified by our classifier?\n
			\tMathematical Formula = True positive/(True positive + False negative)\n
			4) F1-Score:- It is used to combine performance of classifiers(Precision and Recall) into a single metric\n
			It is harmonic mean of precision and recall\n
			\tMathematical Formula = (2 X Precision X Recall)/(Precision+Recall)\n
			''')


	else:
		st.subheader("Predictions")
		st.warning('''
			1) Enter your Tweet.\n
			2) Predict the Label(Language) of the Tweet.\n
			3) Predict the Sentiment of the tweet.\n
			4) Print the percentage of the Prediction.\n 
			''')

########################################## DATA-SET ##########################################

def load_dataset(url):
	DATASET_COLUMNS = ["target","text"]

	# Data is encoded by UTF-8
	DATASET_ENCODING = "ISO-8859-1"

	df = pd.read_csv(url, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

	# adding a new col "length" contains the length of the text
	df['length'] = df['text'].apply(len)

	return df


########################################## Data Description ##########################################
def datadesc(df):
	st.image("image/sent.png", use_column_width = True)

	st.header("DATA-SET Overview")

	if(st.checkbox("About the Data-set")):
		st.info( ''' Social media has opened a whole new world for people around the globe People are just a click away from getting huge chunk of information. \
	  	With information comes people’s opinion and with this comes the positive and negative outlook of people regarding a topic\
	  	Sometimes this also results into bullying and passing on hate comments about someone or something.''')
		st.info('''Formally, given a training sample of tweets and labels, where label ‘4’ denotes the tweet is POSITIVE and label ‘0’ denotes the tweet is NEGATIVE,our objective is to predict the sentiments of the given tweets.''')
		st.success('''text : The tweets collected from various sources and having either positive or negative sentiments associated with it.''')
		st.success('''target : A tweet with label ‘4’ is of positive sentiment while a tweet with label ‘0’ is of negative sentiment.''')
	
	# display the whole dataset
	if(st.checkbox("Show Dataset")):
		h_t=st.selectbox("",('COMPLETE','TOP','BOTTOM'))
		if h_t =='TOP':
			num = st.slider('Select the number of rows', 1,100,5)
			st.write(df.head(num))
		elif h_t=='COMPLETE':
			st.write(df)
		else:
			num = st.slider('Select the number of rows', 1,100,5)
			st.write(df.tail(num))

    # Show shape
	if(st.checkbox("Display the shape")):
		dim = st.selectbox("Rows/Columns?", ("Rows", "Columns"))
		if(dim == "Rows"):
			st.write("Number of Rows", df.shape[0])
		if(dim == "Columns"):
			st.write("Number of Columns", df.shape[1])

    # show value counts
	if(st.checkbox('Counts of Unique values')):
		st.table(df.target.value_counts())

    # show info    
	if(st.checkbox("Show the Data-set Description")):
		st.table(df.describe(include='object'))


########################################## Data Visualization ##########################################
def EDA(df):
	choice=st.radio("Choose one of below graph: ",('Bar Graph','Pie Chart','Histogram','Word Cloud'))

	if choice=='Bar Graph':
		plt.title("CountPlot/Bar Graph")
		st.write(sns.countplot(x='target',data=df))
		st.pyplot(plt)

	elif choice=='Pie Chart':
		plt.title("Pie Chart")
		plt.pie(x=df.target.value_counts(),labels=['Negative','Positive'],explode = [0, 0.1],autopct = '%1.1f%%',shadow=True)
		st.pyplot(plt)

	elif choice=='Histogram':
		st.subheader("Histogram of 'Negative' and 'Positive' tweets with respect to Length")
		df.hist(column='length',by='target')
		st.pyplot(plt)

	# WORD-CLOUD
	else:
	    st.subheader("Treating 'POSITIVE / NEGATIVE' sentiments")
	    dim = st.radio("Positive/Negative?", ("+ tweet", "- tweet"))
	    if(dim == "+ tweet"):
	        df_pos = df.loc[df['target']==4, :]
	        st.write(df_pos.head())
	        st.image("image/WC_positive.png",use_column_width = True)

	    elif(dim == "- tweet"):
	        df_neg = df.loc[df['target']==0, :]
	        st.write(df_neg.head())
	        st.image("image/WC_negative.png", use_column_width = True)


########################################## ML models ##########################################
def Model():

	choice=st.radio("Enter your ML model: ",('Logistic Regression','Decision Tree','Random Forest'))
	
	if choice=="Logistic Regression":
		st.header("Logistic Regression")
		st.subheader("Confusion Matrix")
		st.image("image/lr confusion.png", use_column_width = True)
		st.subheader("Precision Table")
		st.image("image/LR.png", use_column_width = True)
		
	elif choice=="Decision Tree":
		st.header("Decision Tree")
		st.subheader("Confusion Matrix")
		st.image("image/DT confusion.png", use_column_width = True)
		st.subheader("Precision Table")
		st.image("image/DT.png", use_column_width = True)
		
	else:
		st.header("Random Forest")
		st.subheader("Confusion Matrix")
		st.image("image/RF confusion.png", use_column_width = True)
		st.subheader("Precision Table")
		st.image("image/RF.png", use_column_width = True)
		


########################################## Data cleaning ##########################################
def preprocess(raw_tweet):
    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet)
    
    # change sentence to lower case
    letters_only = letters_only.lower()

    # tokenize into words
    words = letters_only.split()
    
    # remove stop words                
    words = [w for w in words if not w in stopwords.words("english")]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    clean_sent = " ".join(words)
    
    return clean_sent


########################################## funcions for Processing ##########################################
def predict(tweet):
    
	# Loading pretrained CountVectorizer from pickle file
	vectorizer = load(open('Model/countvectorizer.pkl', 'rb'))

	# Loading pretrained logistic classifier from pickle file
	classifier = load(open('Model/logit_model.pkl', 'rb'))

	# Preprocessing the tweet
	clean_tweet = preprocess(tweet)

	# Converting text to numerical vector
	clean_tweet_encoded = vectorizer.transform([clean_tweet])

	# Converting sparse matrix to dense matrix
	tweet_input = clean_tweet_encoded.toarray()

	# Prediction
	prediction = classifier.predict(tweet_input)
	pred_prob = classifier.predict_proba(tweet_input)

	if tweet!='':
		if(prediction == 0):
			s_n=round(float(pred_prob[0][0])*100)
			st.warning("Negative Sentiment")
			st.success("Accuracy : "+str(s_n))
			st.image("image/thum_down.png")
		else:
			s_p=round(float(pred_prob[0][1])*100)
			st.info("Positive Sentiment")
			st.success("Accuracy : "+str(s_p))
			st.image("image/thum_up.png")

########################################## Funtion that detect the Language ##########################################

def label(tweet):
    translator = Translator()
    regex = re.compile('[@!#$%^&*()=<>?/\|}{~:]') 
    english=0
    hindi=0
    c="+ univ"
    if(regex.search(tweet) == None):
        c=""
    l=tweet.split() 
    language=['bn','gu','ml','pa','sd','ta','te','ur','hi']

    for i in l:
        t=(translator.translate(i))
        if t.src=='en':
            english=1
        
        elif (t.src in language):
            hindi=1
        
        else :
        	english=1
      

    if english==1 and hindi==1:
        c="Mixed "+c
    elif english==0 and hindi==1:
        c="Hindi "+c
    elif english==1 and hindi==0:
        c="English "+c

    if tweet!='':
    	st.info(c)

########################################## covert the language ##########################################

def convert(tweet):
    translator = Translator()
    t=(translator.translate(tweet))
    return t.text
	
########################################## Final Prediction ##########################################
def Prediction():
	st.image("image/prediction.jpg", use_column_width = True)
	tweet = st.text_input('Enter your tweet')
	label(tweet)
	new_tweet=convert(tweet)
	p=predict(new_tweet)


########################################## main function ##########################################
def main():

	st.title("Twitter Sentiment Analysis")
	choice=load_sidebar()
	df=load_dataset('Rtweet.csv')
	if choice=='Data Description':
		datadesc(df)
	elif choice=='Exploratory Data Analysis':
		EDA(df)
	elif choice=='Model Evaluation':
		Model()
	elif choice=='Predictions':
		Prediction()
	else:
		help_bar()

########################################## Calling Of The App ##########################################
if(__name__=='__main__'):
	main()

