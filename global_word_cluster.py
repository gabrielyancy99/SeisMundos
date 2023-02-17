import Levenshtein as lv
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from google.cloud import translate
import pandas as pd
from transliterate import translit, get_available_language_codes
from unihandecode import Unihandecoder
from translit_me.transliterator import transliterate as tr
from translit_me.lang_tables import *
from ai4bharat.transliteration import XlitEngine
from polyglot.transliteration import Transliterator
from polyglot.downloader import downloader
import geopandas as geo
import os

# COMMAND LINE
os.system("export GOOGLE_APPLICATION_CREDENTIALS='./credentials/sixth-flag-377423-42563eb55ebc.json'")

languages = ["English", "Spanish", "Farsi", "French", "Portuguese", "German", "Swedish", "Norwegian", "Arabic", 
    "Latin", "Polish", "Russian", "Romanian", "Afrikaans", "Albanian", "Bengali", "Bulgarian", 
    "Chinese", "Croatian","Danish","Dutch","Tagalog","Finnish","Italian","Swahili",
    "Indonesian","Kazakh","Macedonian","Hindi","Mongolian","Luxembourgish","Hebrew","Irish",
    "Thai","Lao","Vietnamese","Korean","Ukrainian","Belarusian","Malay","Serbian","Japanese",
    "Hungarian","Greek","Turkish","Slovenian","Slovak","Amharic","Bosnian", "Armenian",
    "Azerbaijani","Czech","Estonian","Georgian","Icelandic","Khmer","Kyrgyz","Latvian","Lithuanian",
    "Nepali","Somali","Tajik","Tamil","Turkmen","Urdu","Uzbek","Sanskrit"]
lang_codes = ["en","es","fa","fr","pt","de","sv","no","ar",
    "la","pl","ru","ro","af","sq","bn","bg","zh","hr","da","nl","fil","fi","it","sw",
    "id","kk","mk","hi","mn","lb","he","ga","th","lo","vi","ko","uk","be","ms","sr","ja",
    "hu","el","tr","sl","sk","am","bs","hy","az","cs","et","ka","is","km","ky","lv","lt",
    "ne","so","tg","ta","tk","ur","uz","sa"]

bdecoder = XlitEngine(beam_width=4, rescore=False, src_script_type="indic")

# Initialize Translation client
def translate_text(text="red", project_id="sixth-flag-377423", output_lang="fr"):
    
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    # breakpoint()
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": output_lang,
        }
    )
    # breakpoint()
    # List of Languages requiring transliteration
    requires_easy_translit = ["hy","mn","el","ru","uk","sr","mk","bg","ka","ky","kk","be"]
    requires_russian = ["ru","be","ky","kk"]
    requires_unicode_translit = ["zh","ja","ko"]
    requires_southasian_translit = ["hi","ne","sa","ta","ur","bn"]
    requires_middleeast_translit = ["ar","he"]
    requires_remaining = ["fa","th","tr","az","km"]#,"am"]
    if output_lang == "es":
        for extension in requires_remaining:
            downloader.download("transliteration2."+extension)
    # Display the translation for each input text provided
    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))
        if output_lang in requires_easy_translit:
            if output_lang in requires_russian:
                transliterated_response = translit(translation.translated_text, "ru", reversed=True)
                print("Transliterated text: {}".format(transliterated_response))
                return transliterated_response.lower()
            else:
                transliterated_response = translit(translation.translated_text, output_lang, reversed=True)
                print("Transliterated text: {}".format(transliterated_response))
                return transliterated_response.lower()
        if output_lang in requires_unicode_translit:
            if output_lang == "ko":
                decoder = Unihandecoder(lang="kr")
            else:
                decoder = Unihandecoder(lang=output_lang)
            print("Transliterated text: {}".format(decoder.decode(translation.translated_text)))
            return decoder.decode(translation.translated_text).lower()
        if output_lang in requires_southasian_translit:
            # breakpoint()
            output = bdecoder.translit_word(translation.translated_text, lang_code=output_lang, topk=1)[0]
            print("Transliterated text: {}".format(output))
            return output
        if output_lang in requires_middleeast_translit:
            if output_lang == "ar":
                print("Transliterated text: {}".format(''.join(tr(translation.translated_text, AR_EN))))
                return ''.join(tr(translation.translated_text, AR_EN)).lower()
            if output_lang == "he":
                print("Transliterated text: {}".format(''.join(tr(translation.translated_text, HE_EN))))
                return ''.join(tr(translation.translated_text, HE_EN)).lower()
        if output_lang in requires_remaining:
            transliterator = Transliterator(source_lang=output_lang, target_lang="en")
            prompt = transliterator.transliterate(translation.translated_text)
            print("Transliterated text: {}".format(prompt))
            return prompt
        else:
            return translation.translated_text.lower()

def cluster_words(dataframe, num_clusters=3, cluster_alg="ratio"):
    list_of_words = dataframe[0]
    similarityMatrix = np.zeros((np.size(list_of_words),np.size(list_of_words)))

    for i in np.arange(np.size(list_of_words)):
        for j in np.arange(np.size(list_of_words)):
            if cluster_alg == "ratio":
                similarityMatrix[i,j] = lv.ratio(list_of_words[i],list_of_words[j])
            if cluster_alg == "distance":
                similarityMatrix[i,j] = lv.distance(list_of_words[i],list_of_words[j])

    print(similarityMatrix)

    # Perform clustering, you can choose the method
    # in this case, we use 'ward'
    Z = linkage(similarityMatrix, 'ward')

    # Extract the membership to a cluster, either specify the n_clusters
    # or the cut height
    # (similar to sklearn labels)
    print(cut_tree(Z, n_clusters=num_clusters))

    # Add Cluster group Labels
    group_labels = cut_tree(Z, n_clusters=num_clusters)

    # data_frame = pd.DataFrame([list_of_words,group_labels])#,columns=['translated_word','group_labels'])
    # data_frame = data_frame.T
    dataframe['group_labels'] = group_labels

    # # Visualize the clustering as a dendogram
    fig = plt.figure(figsize=(25, 16))
    plt.title(dataframe[0][0], fontdict={'fontsize' : 20});
    dn = dendrogram(Z, orientation='right', labels=list(dataframe['label']))
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=12)
    # Remove x-ticks
    plt.xticks([]);

    # Remove the outer borders
    plt.gca().spines['top'].set_visible(False);
    plt.gca().spines['right'].set_visible(False);
    plt.gca().spines['bottom'].set_visible(False);
    plt.savefig('dendrogram_ex.png');
    # plt.show();

    return dataframe

def make_world_plot(data):
    # Set the plot size for this notebook:
    world = geo.read_file(geo.datasets.get_path('naturalearth_lowres'))
    # ax = world.plot()
    plt.rcParams["figure.figsize"]=20,12

    # Set Languages for Countries
    world['language'] = np.zeros(177)
    world.loc[world[world.name == 'Bahamas'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Belize'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Benin'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Botswana'].index[0],'language'] = 'English'
    world.loc[world[world.name == "Cameroon"].index[0],'language'] = 'English'
    world.loc[world[world.name == "Côte d'Ivoire"].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Dem. Rep. Congo'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Falkland Is.'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Fiji'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Fr. S. Antarctic Lands'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Ghana'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Greenland'].index[0],'language'] = 'Greenlandic'
    world.loc[world[world.name == 'Guinea'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Guinea-Bissau'].index[0],'language'] = 'Portuguese'
    world.loc[world[world.name == 'Guyana'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Indonesia'].index[0],'language'] = 'Indonesian'
    world.loc[world[world.name == 'Kazakhstan'].index[0],'language'] = 'Kazakh'
    world.loc[world[world.name == 'Lesotho'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Liberia'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Mali'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Mauritania'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Namibia'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Niger'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Nigeria'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Papua New Guinea'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'South Africa'].index[0],'language'] = 'Afrikaans'
    world.loc[world[world.name == 'Afghanistan'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Algeria'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Argentina'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Australia'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Bolivia'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Brazil'].index[0],'language'] = 'Portuguese'
    world.loc[world[world.name == 'Canada'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Chad'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Chile'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'China'].index[0],'language'] = 'Chinese'
    world.loc[world[world.name == 'Colombia'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Costa Rica'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Cuba'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Czechia'].index[0],'language'] = 'Czech'
    world.loc[world[world.name == 'Dominican Rep.'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Ecuador'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Egypt'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'El Salvador'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Ethiopia'].index[0],'language'] = 'Amharic'
    world.loc[world[world.name == 'France'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Finland'].index[0],'language'] = 'Finnish'
    world.loc[world[world.name == 'Germany'].index[0],'language'] = 'German'
    world.loc[world[world.name == 'Guatemala'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Haiti'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Honduras'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Iran'].index[0],'language'] = 'Farsi'
    world.loc[world[world.name == 'Iraq'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Italy'].index[0],'language'] = 'Italian'
    world.loc[world[world.name == 'Jamaica'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Japan'].index[0],'language'] = 'Japanese'
    world.loc[world[world.name == 'Jordan'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Kenya'].index[0],'language'] = 'Swahili'
    world.loc[world[world.name == 'Lebanon'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Libya'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'North Macedonia'].index[0],'language'] = 'Macedonian'
    world.loc[world[world.name == 'Mexico'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Morocco'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Nicaragua'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Norway'].index[0],'language'] = 'Norwegian'
    world.loc[world[world.name == 'Oman'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Palestine'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Panama'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Paraguay'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Peru'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Poland'].index[0],'language'] = 'Polish'
    world.loc[world[world.name == 'Puerto Rico'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Qatar'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Russia'].index[0],'language'] = 'Russian'
    world.loc[world[world.name == 'Portugal'].index[0],'language'] = 'Portuguese'
    world.loc[world[world.name == 'Saudi Arabia'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Senegal'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Serbia'].index[0],'language'] = 'Serbian'
    world.loc[world[world.name == 'Slovakia'].index[0],'language'] = 'Slovak'
    world.loc[world[world.name == 'Slovenia'].index[0],'language'] = 'Slovenian'
    world.loc[world[world.name == 'Somalia'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'S. Sudan'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Spain'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Sudan'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Syria'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Sweden'].index[0],'language'] = 'Swedish'
    world.loc[world[world.name == 'Tanzania'].index[0],'language'] = 'Swahili'
    world.loc[world[world.name == 'Timor-Leste'].index[0],'language'] = 'Portuguese'
    world.loc[world[world.name == 'Uzbekistan'].index[0],'language'] = 'Uzbek'
    world.loc[world[world.name == 'United Arab Emirates'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'United Kingdom'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'United States of America'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Uruguay'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Venezuela'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Vietnam'].index[0],'language'] = 'Vietnamese'
    world.loc[world[world.name == 'Yemen'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Angola'].index[0],'language'] = 'Portuguese'
    world.loc[world[world.name == 'Burkina Faso'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Burundi'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Cambodia'].index[0],'language'] = 'Khmer'
    world.loc[world[world.name == 'Central African Rep.'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Congo'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Eq. Guinea'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Gabon'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Gambia'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'India'].index[0],'language'] = 'Hindi'
    world.loc[world[world.name == 'Israel'].index[0],'language'] = 'Hebrew'
    world.loc[world[world.name == 'Kuwait'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Laos'].index[0],'language'] = 'Lao'
    world.loc[world[world.name == 'Madagascar'].index[0],'language'] = 'French'
    world.loc[world[world.name == 'Malawi'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Mongolia'].index[0],'language'] = 'Mongolian'
    world.loc[world[world.name == 'Mozambique'].index[0],'language'] = 'Portuguese'
    world.loc[world[world.name == 'Myanmar'].index[0],'language'] = 'Burmese'
    world.loc[world[world.name == 'North Korea'].index[0],'language'] = 'Korean'
    world.loc[world[world.name == 'Sierra Leone'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'South Korea'].index[0],'language'] = 'Korean'
    world.loc[world[world.name == 'Suriname'].index[0],'language'] = 'Dutch'
    world.loc[world[world.name == 'Thailand'].index[0],'language'] = 'Thai'
    world.loc[world[world.name == 'Albania'].index[0], 'language'] = 'Albanian'
    world.loc[world[world.name == 'Armenia'].index[0], 'language'] = 'Armenian'
    world.loc[world[world.name == 'Austria'].index[0], 'language'] = 'German'
    world.loc[world[world.name == 'Bangladesh'].index[0], 'language'] = 'Bengali'
    world.loc[world[world.name == 'Belarus'].index[0], 'language'] = 'Belarusian'
    world.loc[world[world.name == 'Bhutan'].index[0], 'language'] = 'Dzongkha'
    world.loc[world[world.name == 'Bulgaria'].index[0], 'language'] = 'Bulgarian'
    world.loc[world[world.name == 'Croatia'].index[0], 'language'] = 'Croatian'
    world.loc[world[world.name == 'Estonia'].index[0], 'language'] = 'Estonian'
    world.loc[world[world.name == 'Greece'].index[0], 'language'] = 'Greek'
    world.loc[world[world.name == 'Hungary'].index[0], 'language'] = 'Hungarian'
    world.loc[world[world.name == 'Kyrgyzstan'].index[0], 'language'] = 'Kyrgyz'
    world.loc[world[world.name == 'Latvia'].index[0], 'language'] = 'Latvian'
    world.loc[world[world.name == 'Lithuania'].index[0], 'language'] = 'Lithuanian'
    world.loc[world[world.name == 'Luxembourg'].index[0], 'language'] = 'Luxembourgish'
    world.loc[world[world.name == 'Moldova'].index[0], 'language'] = 'Romanian'
    world.loc[world[world.name == 'Nepal'].index[0], 'language'] = 'Nepali'
    world.loc[world[world.name == 'Azerbaijan'].index[0], 'language'] = 'Azerbaijani'
    world.loc[world[world.name == 'Belgium'].index[0], 'language'] = 'Dutch'
    world.loc[world[world.name == 'Brunei'].index[0], 'language'] = 'Malay'
    world.loc[world[world.name == 'Denmark'].index[0], 'language'] = 'Danish'
    world.loc[world[world.name == 'Eritrea'].index[0], 'language'] = 'Arabic'
    world.loc[world[world.name == 'Georgia'].index[0], 'language'] = 'Georgian'
    world.loc[world[world.name == 'Iceland'].index[0], 'language'] = 'Icelandic'
    world.loc[world[world.name == 'Ireland'].index[0], 'language'] = 'Irish'
    world.loc[world[world.name == 'Malaysia'].index[0], 'language'] = 'Malay'
    world.loc[world[world.name == 'Netherlands'].index[0], 'language'] = 'Dutch'
    world.loc[world[world.name == 'New Caledonia'].index[0], 'language'] = 'French'
    world.loc[world[world.name == 'New Zealand'].index[0], 'language'] = 'English'
    world.loc[world[world.name == 'Pakistan'].index[0], 'language'] = 'Urdu'
    world.loc[world[world.name == 'Philippines'].index[0], 'language'] = 'Tagalog'
    world.loc[world[world.name == 'Romania'].index[0], 'language'] = 'Romanian'
    world.loc[world[world.name == 'Switzerland'].index[0], 'language'] = 'German'
    world.loc[world[world.name == 'Sri Lanka'].index[0], 'language'] = 'Tamil'
    world.loc[world[world.name == 'Taiwan'].index[0], 'language'] = 'Chinese'
    world.loc[world[world.name == 'Tajikistan'].index[0], 'language'] = 'Tajik'
    world.loc[world[world.name == 'Togo'].index[0], 'language'] = 'French'
    world.loc[world[world.name == 'Tunisia'].index[0], 'language'] = 'Arabic'
    world.loc[world[world.name == 'Turkey'].index[0], 'language'] = 'Turkish'
    world.loc[world[world.name == 'Turkmenistan'].index[0], 'language'] = 'Turkmen'
    world.loc[world[world.name == 'Vanuatu'].index[0], 'language'] = 'English'
    world.loc[world[world.name == 'Bosnia and Herz.'].index[0],'language'] = 'Bosnian'
    world.loc[world[world.name == 'Cyprus'].index[0],'language'] = 'Greek'
    world.loc[world[world.name == 'Djibouti'].index[0],'language'] = 'Arabic'
    world.loc[world[world.name == 'Kosovo'].index[0],'language'] = 'Albanian'
    world.loc[world[world.name == 'Montenegro'].index[0],'language'] = 'Serbian'
    world.loc[world[world.name == 'N. Cyprus'].index[0],'language'] = 'Turkish'
    world.loc[world[world.name == 'Rwanda'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Solomon Is.'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Somaliland'].index[0],'language'] = 'Somali'
    world.loc[world[world.name == 'Trinidad and Tobago'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Uganda'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'W. Sahara'].index[0],'language'] = 'Spanish'
    world.loc[world[world.name == 'Zambia'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Zimbabwe'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'eSwatini'].index[0],'language'] = 'English'
    world.loc[world[world.name == 'Ukraine'].index[0],'language'] = 'Ukrainian'

    # Set Colors Of Major Languages to Color Code First Map
    # world[world.language == 'English'].plot(color='blue',ax=ax)
    # world[world.language == 'Spanish'].plot(color='yellow',ax=ax)
    # world[world.language == 'Arabic'].plot(color='green',ax=ax)
    # world[world.language == 'French'].plot(color='orange',ax=ax)
    # world[world.language == 'Russian'].plot(color='red',ax=ax)
    # world[world.language == 'Portuguese'].plot(color='purple',ax=ax)
    # world[world.language == 'Swahili'].plot(color='brown',ax=ax)
    # world[world.language == 'German'].plot(color='indago',ax=ax)

    # Add the Whole Translation Datastructure to the Map
    # world_full = world.merge(datastruct, left_on='language', right_on='language')
    world_full = world.merge(data, left_on='language', right_on=1)
    # Place Words on Centroid of Gemetry for Each Country - Sorted by Language
    world_full['coords'] = world_full['geometry'].apply(lambda x: x.centroid.coords[:])
    world_full['coords'] = [coords[0] for coords in world_full['coords']]
    for language in languages:
        if world_full[world_full.language == language].index is not None:
            for row in world_full[world_full.language == language].index:
                plt.text(s=world_full.loc[row,0], x = world_full.loc[row,'coords'][0], y = world_full.loc[row,'coords'][1],
                        horizontalalignment='center', fontdict = {'size': 12})

    # Remove x and y-ticks
    plt.xticks([]);
    plt.yticks([]);

    # plt.show();
    # plt.close();

    ################# Second Plot With Color Coding by Cluster ID #####################################
    world2 = geo.read_file(geo.datasets.get_path('naturalearth_lowres'))
    ax2 = world2.plot()
    c = list(mcolors.TABLEAU_COLORS)
    for i in np.arange(world_full['group_labels'].max()+1):
        # breakpoint()
        world_full[world_full['group_labels'] == i].plot(color=c[i],ax=ax2)
        for language in languages:
            if world_full[world_full.language == language].index is not None:
                for row in world_full[world_full.language == language].index:
                    plt.text(s=world_full.loc[row,0], x = world_full.loc[row,'coords'][0], y = world_full.loc[row,'coords'][1],
                            horizontalalignment='center', fontdict = {'size': 12})

    # Remove x and y-ticks
    plt.xticks([]);
    plt.yticks([]);
    plt.title('Clustered World Map for '+data[0][0], fontdict={'fontsize' : 20})
    plt.savefig('world_ex.png');
    # plt.show();
    # plt.close();

    ################################# Zooming on Europe ###############################################
    proj = '+proj=mill +lon_0=12'
    # Load the shapefile of the map data
    map_data = world_full
    map_data = map_data.to_crs(proj)
    # breakpoint()
    map_data.loc[61, 'geometry'] = list(map_data.loc[61, 'geometry'].geoms)[1]
    map_data['coords'] = map_data['geometry'].apply(lambda x: x.centroid.coords[:])
    map_data['coords'] = [coords[0] for coords in map_data['coords']]
    map_data.loc[2,'geometry'] = None
    # Set the bounds of the area to display
    xmin, ymin, xmax, ymax = [-3000000, 4000000, 4000000, 10000000]
    # Create a bounding box polygon
    roi = map_data.cx[xmin:xmax, ymin:ymax]
    # breakpoint()
    # roi = map_data
    ax3 = roi.plot()
    for i in np.arange(map_data['group_labels'].max()+1):
        # breakpoint()
        map_data[map_data['group_labels'] == i].plot(color=c[i],ax=ax3)
        for language in languages:
            if map_data[map_data.language == language].index is not None:
                for row in map_data[map_data.language == language].index:
                    if map_data.loc[row,'coords'][0] >= (xmin-500000):
                        if map_data.loc[row,'coords'][0] <= (xmax+500000):
                            if map_data.loc[row,'coords'][1] >= (ymin-500000):
                                    plt.text(s=map_data.loc[row,0], x = map_data.loc[row,'coords'][0], y = map_data.loc[row,'coords'][1],
                                        horizontalalignment='center', fontdict = {'size': 14})
    plt.xlim(-5000000, 4000000)
    plt.ylim(3800000, 10000000)

    # Remove x and y-ticks
    plt.xticks([]);
    plt.yticks([]);
    plt.title('Clustered Map of Europe for '+data[0][0], fontdict={'fontsize' : 20})
    plt.savefig('europe_ex.png');
    # plt.show()

    ############################### Zooming on Asia ############################################
    map_data_asia = map_data
    # Set the bounds of the area to display
    xmin, ymin, xmax, ymax = [1000000, -2800000, 20000000, 8000000]
    # Create a bounding box polygon
    roi2 = map_data_asia.cx[xmin:xmax, ymin:]
    # breakpoint()
    # roi = map_data
    ax4 = roi2.plot()
    for i in np.arange(map_data_asia['group_labels'].max()+1):
        # breakpoint()
        map_data_asia[map_data_asia['group_labels'] == i].plot(color=c[i],ax=ax4)
        for language in languages:
            if map_data_asia[map_data_asia.language == language].index is not None:
                for row in map_data_asia[map_data_asia.language == language].index:
                    if map_data_asia.loc[row,'coords'][0] >= (xmin-500000):
                        if map_data_asia.loc[row,'coords'][0] <= (xmax+500000):
                            if map_data_asia.loc[row,'coords'][1] >= (ymin-500000):
                                if map_data.loc[row,'coords'][1] <= (ymax-500000):
                                    plt.text(s=map_data_asia.loc[row,0], x = map_data_asia.loc[row,'coords'][0], y = map_data_asia.loc[row,'coords'][1],
                                        horizontalalignment='center', fontdict = {'size': 14})
    plt.xlim(1000000, 20000000)
    plt.ylim(-2800000, 8000000)

    # Remove x and y-ticks
    plt.xticks([]);
    plt.yticks([]);
    plt.title('Clustered Map of Asia for '+data[0][0], fontdict={'fontsize' : 20})
    plt.savefig('asia_ex.png');
    # plt.show()


def run_main(term, num_clust, method):
    if os.path.isfile('unclustered_words.npz'):
        with open('unclustered_words.npz','rb') as f:
            data_frame = np.load(f, allow_pickle=True)
            # data_frame = pd.DataFrame(data_frame, columns=['translation','language','lang_code','label'])
            data_frame = pd.DataFrame(data_frame, columns=[0,1,2,'label'])
        if data_frame[0][0] == term:
            #Perform Clustering and Compile DataFrame
            datastruct = cluster_words(data_frame, num_clusters=num_clust, cluster_alg=method)
            make_world_plot(datastruct)
        else:
            all_langs = [term]
            for lang_code in lang_codes[1:]:
                all_langs.append(translate_text(text=term, output_lang=lang_code))

            #Compile Dataframe
            data_frame = pd.DataFrame([all_langs,languages,lang_codes])
            data_frame = data_frame.T
            data_frame['label'] = data_frame[1]+' - '+data_frame[0]
            with open('unclustered_words.npz','wb') as f:
                np.save(f, data_frame)

            #Perform Clustering and Compile DataFrame
            datastruct = cluster_words(data_frame, num_clusters=num_clust, cluster_alg=method)

            make_world_plot(datastruct)
    else:        
        all_langs = [term]
        for lang_code in lang_codes[1:]:
            all_langs.append(translate_text(text=term, output_lang=lang_code))

        #Compile Dataframe
        data_frame = pd.DataFrame([all_langs,languages,lang_codes])
        data_frame = data_frame.T
        data_frame['label'] = data_frame[1]+' - '+data_frame[0]
        with open('unclustered_words.npz','wb') as f:
            np.save(f, data_frame)

        #Perform Clustering and Compile DataFrame
        datastruct = cluster_words(data_frame, num_clusters=num_clust, cluster_alg=method)

        make_world_plot(datastruct)