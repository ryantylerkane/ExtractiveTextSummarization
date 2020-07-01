import sys
import os
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
import rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import string
from nltk.translate.bleu_score import SmoothingFunction
import csv


# Comparison of open source text extraction-based summarization algorithms (LexRank, TextRank, and Latent Semantic Analysis) using BLEU and ROGUE scores.
# By: Ryan Kane
# 12/2/9

#Each of the included libraries tend to keep their own list of English stop words, which may result in different words being considered for
#summaries and scores in each algorithm. We can extract our own stop words into an array and feed it into each algorithm in order for the same
#terms to be considered significant regardless of the algorithm being applied.

fileCounter = 0 #Global file counter that will be used to calculate the average value for BLEU-4, ROUGE-1, and ROUGE-L.


textRankAVG = [0.00]*3 #Declare an empty array of size 3. Index 0 will hold the sum of Rouge-1 F values, index 1 will hold the sum of Rouge-L values, and index 2 will hold sum of Bleu-4 values.
lexRankAVG = [0.00]*3 #Declare an empty array of size 3. Index 0 will hold the sum of Rouge-1 F values, index 1 will hold the sum of Rouge-L values, and index 2 will hold sum of Bleu-4 values.
lsaAVG = [0.00]*3 #Declare an empty array of size 3. Index 0 will hold the sum of Rouge-1 F values, index 1 will hold the sum of Rouge-L values, and index 2 will hold sum of Bleu-4 values.

def loadStopwords(): #Function reads the list of stop words from the text file in the project directory and saves each term to an array.
    inputHandler = open("stopwords.txt", 'r')
    stopwords = []
    for word in inputHandler:
        newWord = word.replace('\n','')
        stopwords.append(newWord)
    inputHandler.close()
    return stopwords

def readFiles(stopWords,outputHandler, textRankHandler, lexRankHandler, lsaHandler): #For each file in the data set, the function separates the gold summary from the article, and then passes the components to other functions for summarization and scoring.
   directory = sys.argv[1]  
   tests = ['TEXTRANK', 'LEXRANK', 'LSA']
   spaceyObj = spacy.load('en_core_web_sm') #Open a new spacy object to count the number of sentences in each summary.
   for file in os.listdir(directory):
       try:
    #The label ABSTRACT representing the summary is contained alone on the first line, so we can ignore that first line.
        inputHandler = open(directory + "\\"+ file)
        lines = inputHandler.readlines() #Extract all lines from the file.
        goldSummary = lines[1] #Place the summary in a new string. Summary [2] should hold the label "INTRODUCTION" representing the beginning of the actual article. Indices from 3 to the end should represent article text.
        summarySentenceCount = countSentences(goldSummary,spaceyObj) #Obtain the number of sentences found in the abstract so we can attempt to create a summary of our own with the same number.
        paragraphs = lines[3:] #Remove the labels and the summary so we obtain an array only containing article text.
        article = " ".join(paragraphs) #Rejoin the pargraphs to one string so that they can be analyzed by the summarizing algorithms.
        outputHandler.write("FILE NAME: " + file)
        global fileCounter
        fileCounter = fileCounter+1 #Increase the number of files counted.
        parser = PlaintextParser.from_string(article, Tokenizer("english")) #Parser object required for sumy summarization objects.
        global lsaAVG
        global textRankAVG
        global lexRankAVG
        for test in tests: #Obtain handler in order to record scores to the correct file. Also obtain a pointer to the global array that will be used to hold running total for average calculations.
            if test=="LSA":
                handler = lsaHandler
                averageArray = lsaAVG
            elif test=="TEXTRANK":
                handler = textRankHandler
                averageArray = textRankAVG
            else:
                handler = lexRankHandler
                averageArray = lexRankAVG
            rowBuilder = [] #Will hold row that will be entered into output csv file.
            machineSummary= getSummary(test, summarySentenceCount, stopWords, parser, outputHandler) #Obtain the summary using the method represented by the string "test"
            rowBuilder = rougeScore(goldSummary, machineSummary, rowBuilder, averageArray) #Add ROUGE scores to row intended for the output file.
            rowBuilder = bleuScore(goldSummary, machineSummary, stopWords, rowBuilder, averageArray) #Add BLEU scores to row intended for output file.
            handler.writerow({'FILE_NAME': file, 'ROUGE-1': rowBuilder[0], 'ROUGE-2': rowBuilder[1], 'ROUGE-3': rowBuilder[2], 'ROUGE-4': rowBuilder[3], 'ROUGE-L': rowBuilder[4], 'BLEU-1': rowBuilder[5], 'BLEU-2': rowBuilder[6], 'BLEU-3': rowBuilder[7], 'BLEU-4': rowBuilder[8]})
        outputHandler.write("\n\n") #Skip lines for formatting.
       except Exception as ex:
               raise ex
               print("Files unable to be read in directory: " + directory)

def countSentences(summary, spaceyObj): #Function counts the number of sentences in a summary extracted from the dataset.
    documentCount = spaceyObj(summary) #Use spacy object to count the number of sentence in a sumamry.
    return len(list(documentCount.sents))

def getSummary(testName, summarySentenceCount, stopWords, parser,outputHandler): #Function generates an object for whichever algorithm is found in the testName string and then writes the summary produced by that algorithm to the output file.
    if testName =="LSA":
        summarizer = LsaSummarizer()
    elif testName =="TEXTRANK":
        summarizer = TextRankSummarizer()
    else:
        summarizer = LexRankSummarizer()

    summarizer.stop_words = stopWords #Use the custom list of stop words extracted.
    summary = summarizer(parser.document, summarySentenceCount) #Obtain the summary using the method provided in the testName argument.

    words=[]

    outputHandler.write("\n" + testName + " SUMMARY: ")
    for s in summary:
        outputHandler.write(str(s))
        words.append(str(s))
    outputHandler.write("\n")
    finalSummary = " ".join(words) #Join each sentence into one string so it can be used to generate a score.
    return finalSummary


def rougeScore(goldSummary, machineSummary, rowBuilder, averageArray): #Function generates ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-4, and ROUGE-L scores given a gold and machine-generated summary.
    #Create a new Rogue object using the default settings provided in documentation.
    rogueScores = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=4, limit_length=False, apply_avg=False, apply_best=False, alpha=0.5, weight_factor=1.2, stemming=True) #Use default parameters recommended by library developer. 
    scores=rogueScores.get_scores(machineSummary,goldSummary) #Generate the n-gram and rouge-l scores.
    rowBuilder.append(scores.get('rouge-1')[0].get('f')[0]) #Add ROUGE-1 F1 value to the row.
    rowBuilder.append(scores.get('rouge-2')[0].get('f')[0]) #Add ROUGE-2 F1 value to the row.
    rowBuilder.append(scores.get('rouge-3')[0].get('f')[0]) #Add ROUGE-3 F1 value to the row.
    rowBuilder.append(scores.get('rouge-4')[0].get('f')[0]) #Add ROUGE-4 F1 value to the row.
    rowBuilder.append(scores.get('rouge-l')[0].get('f')[0]) #Add ROUGE-L F1 value to the row.

    #Add to existing running totals for average calculations.
    averageArray[0] = averageArray[0] + scores.get('rouge-1')[0].get('f')[0]
    averageArray[1] = averageArray[1] + scores.get('rouge-l')[0].get('f')[0]

    return rowBuilder

def bleuScore(goldSummary, machineSummary, stopWords, rowBuilder, averageArray): #Function generates cumulative BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores for a provided gold and machine-generated summary.
    #We must tokenize the words in each summary in order for the library to be able to calculate the BLEU scores.
    #We will use our custom list of stop words to remove words as they split into tokens.
    goldWords = []
    summaryWords = []

    goldTokens = word_tokenize(goldSummary) #Extract the tokens from the gold summary.
    summaryTokens = word_tokenize(machineSummary) #Extract tokens from the machine generated summary.

    for word in goldTokens:
       if word not in stopWords and word not in string.punctuation:
           goldWords.append(word) #Only append the token to the list to be analyzed if it is not a stop word or a punctuation mark.
    for w in summaryTokens:
        if w not in stopWords and w not in string.punctuation:
            summaryWords.append(w) #Only append the token to the list to be analyzed if it is not a stop word or a punctuation mark.
    
    rowBuilder.append(corpus_bleu([[goldWords]], [summaryWords], weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=SmoothingFunction().method7)) #Generate BLEU-1 score.
    rowBuilder.append(corpus_bleu([[goldWords]], [summaryWords], weights=(0.50, 0.50, 0.00, 0.00), smoothing_function=SmoothingFunction().method7)) #Generate BLEU-2 score.
    rowBuilder.append(corpus_bleu([[goldWords]], [summaryWords], weights=(0.33, 0.33, 0.33, 0.00), smoothing_function=SmoothingFunction().method7)) #Generate BLEU-3 score weighing unigrams, bigrams, and trigrams evenly. Use smoothing for trigrams that do not exist.
    bleu4 = corpus_bleu([[goldWords]], [summaryWords], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method7)
    rowBuilder.append(bleu4) #Generate the BLEU-4 score weighing unigrams, bigrams, trirgrams, and 4-grams equally. Use smoothing to compensate for 3-gram and 4-grams that do not exist.
    
    #Add to running total for average calculation.
    averageArray[2] = averageArray[2] + bleu4 #Add BLEU-4 score for future average calculations.

    return rowBuilder

def calculateAverages(resultHandler): #Uses global variables to calculate the average ROUGE-1, ROUGE-L, BLEU-4, and combined BLEU-4 & ROUGE-L scores. score for each summarization algorithm.
    global fileCounter
    global textRankAVG
    global lexRankAVG
    global lsaAVG

    avgs = [textRankAVG, lexRankAVG, lsaAVG]
    names = ['TEXTRANK', 'LEXRANK', 'LSA']

    for i in range(len(names)):
        resultHandler.write(names[i] + " ROUGE-1 AVG: " + str(avgs[i][0]/fileCounter) + "\n")
        rougelout = avgs[i][1]/fileCounter
        resultHandler.write(names[i] + " ROUGE-L AVG: " + str(rougelout) + "\n")
        bleu4out = avgs[i][2]/fileCounter
        resultHandler.write(names[i] + " CUMULATIVE BLEU-4 AVG: " + str(bleu4out) + "\n")
        resultHandler.write(names[i] + " COMBINED ROUGE-L AND BLEU-4 F1 SCORE: " + str(2*(bleu4out*rougelout)/(bleu4out+rougelout)) + "\n")
        resultHandler.write("\n\n")

stopWords = loadStopwords() #Obtain stop words from the provided text file.
outputHandler = open("Summaries.txt", 'w') #Open new file to record all machine-generated summaries.

#Initialize CSV files that will hold raw BLEU and ROUGE scores and file that will show ROUGE-1, ROUGE-L, and BLEU-4 averages.
textRankFile = open("TextRankScores.csv", 'w')
textRankHandler = csv.DictWriter(textRankFile, ['FILE_NAME', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'ROUGE-L', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'], lineterminator='\n')
textRankHandler.writeheader()
lexRankFile = open("LexRankScores.csv", 'w')
lexRankHandler = csv.DictWriter(lexRankFile, ['FILE_NAME', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'ROUGE-L', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'], lineterminator='\n')
lexRankHandler.writeheader()
lsaFile = open("LSAScores.csv", 'w')
lsaHandler = csv.DictWriter(lsaFile, ['FILE_NAME', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'ROUGE-L', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'], lineterminator='\n')
lsaHandler.writeheader()
resultHandler = open("AverageScores.txt", 'w')
readFiles(stopWords, outputHandler, textRankHandler, lexRankHandler, lsaHandler) #Generate and score each summary type for every file.
calculateAverages(resultHandler) #Calculate average ROUGE-1, ROUGE-L, and BLEU-4 scores for each type of summarization.
#Close all files.
outputHandler.close()
textRankFile.close()
lexRankFile.close()
lsaFile.close()
resultHandler.close()