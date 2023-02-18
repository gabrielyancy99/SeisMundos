# SeisMundos - Etymology Approximation with Levenshtein Distance
<figure>
  <img src="/example/seismundos_example.png">
  <figcaption><em>Screenshot of program results for the word yellow with 6 clusters. (See example folder for full size graph outputs.)</em></figcaption>
</figure>

# Brief Description

SeisMundos is a language analysis application that uses the Levenshtein distance algorithm to find the etymology of a given English word across different languages. The application uses the Global Word Cluster (GWC) library to generate clusters of languages based on their similarity. The user can enter an English word and the desired number of clusters, and the application will generate a graph showing the distribution of the word across languages. The user can also save the graph and reopen it in a separate window for further analysis.

The application is built using PyQt6, which provides a graphical user interface for the application. The main window contains entry fields for the search term and cluster number, as well as options for selecting the Levenshtein distance or another distance metric. The main window displays the graph and options for saving and reopening the graph in a separate window.

# Running the Application

## Installation Instructions

To run the application locally first clone the repository to your personal machine. From the project folder open CMD/Terminal and run the command below:

`pip install --upgrade --user -r requirements.txt`

To open the preconfigured application built with PyQt enter:

`python SeisMundos_GUI.py` | `python3 SeisMundos_GUI.py` | double click `SeisMundos_GUI.py`

## Program Description

Enter in the word you want to analyze in English as well as the number of clusters you want to view and the method you wish to use for looking at similarity between words. Then hit the button to generate plots. Four plots are generated each time: a dendrogram, a world map with cluster color coding, a map zoomed in on Europe, and a map zoomed in on Asia. If you close any of the graphs, you can reopen them by double clicking on the graph from the main application window. You can also hit the 'Save Graph' button on any of the graph windows to ensure a copy of the graph is stored locally before running the application again. If you don't save a copy, the local version of the graph will be rewritten when the application is run again. 

###### What's Happening Under the Hood

> The copy will be in the same folder as the repository and will have a name with an added long numerical extension indicating the last modified time of the copied grpah. This is done because each run of the application will have a unique time, therefore each saved/copied graph will automatically have a unique filename. Each time you run the program a copy of the graphs are stored locally, but these copies are overwritten each time to conserve memory.

## Repository Description

The run_main function, found in the Global Word Cluster package, or global_word_cluster.py, is one of the main functions of the project. It takes three arguments: term, which is the word to be clustered, num_clust, which is the desired number of clusters, and method, which is the clustering algorithm to be used. The function first checks if there is a file called 'unclustered_words.npz' that contains unclustered words. If the file exists, it loads the file as a DataFrame and checks if the first word in the DataFrame is the same as the input term. If the word is the same, the function performs clustering using the provided clustering algorithm and compiles a data structure. If the word is different, the function translates the word into other languages and creates a DataFrame with the translated words. Then, it performs clustering and compiles a data structure. In both cases, the function generates a dendrogram and a world plot that shows the clusters on a world map.

The project also includes other functions such as translate_text, which uses the Google Translate API to translate a given text into a specified language, and cluster_words, which performs clustering on a given DataFrame using the specified clustering algorithm. The project is written in Python and uses various open-source libraries, especially for the transliteration of translations provided by the Google Translate API.

# Troubleshooting

If you get an error that reads:

> google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials. Please set GOOGLE_APPLICATION_CREDENTIALS or explicitly create credentials and re-run the application. For more information, please see https://cloud.google.com/docs/authentication/getting-started

Try going into the code for either of the python files and copying the string at the top in os.system(). Run this string, without the outer "", in your command line terminal. Then try to run the application again.
