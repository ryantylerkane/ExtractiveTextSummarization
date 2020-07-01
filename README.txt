The file TextSummarization.py contains all necessary code needed to run the project.
This program runs all summarization algorithms on each article of the DUC corpus and
outputs the resulting summaries in "Summaries.txt." In addition, the program will also
generate a CSV file for each of the three algorithms labeled with the algorithm name followed
by "scores.csv". These files will contain all calculated ROUGE-n, ROUGE-L, and BLEU-N scores 
associated with the summary generated from a particular file name. Last, the program outputs
"Averages.txt", which gives the average ROUGE-1, ROUGE-L, BLEU-4, and combined ROUGE-L/BLEU4
F1 metric for each algorithm type.

How to run the program:
1. Extract TextSummarization.py and stopwords.txt to the same directory. 
2. Extract the DUC2001 folder to any directory of your choice.
3. Download each of the following libraries to Python:
   https://pypi.org/project/sumy/
   https://pypi.org/project/spacy/
   https://pypi.org/project/py-rouge/
   https://pypi.org/project/nltk/
 
   During development this was done through the command line using the recommended
   pip command on each of the libraries' web page.
4. Run TextSummarization.py, specifiying the location of the "Summaries" sub-folder of the
   DUC2001 directory as the lone parameter on the command line. For instance, this program was
   debugged and tested using "C:\Users\Ryan\Desktop\data set\DUC2001\Summaries" as argv[1].
5. Wait for the program to run. The five aforementioned output files will appear in the same
   directory in TextSummarization.py is located.

NOTES:
This project was coded, debugged, and tested using Microsoft Visual Studio 2019 due
to the Spacy library being unable to run without the latest Microsoft Visual 2015-2019 C++ 
Redistrution packages. The program should still run on another interpreter (such as IDLE), 
providing the distribution packages are installed.
