
# coding: utf-8

# In[3]:
from bokeh.models import Panel
from bokeh.models.widgets.panels import Tabs
from collections import Counter
import string, unicodedata
import pandas as pd
from bokeh.plotting import figure
from nltk.util import ngrams
from bokeh.palettes import viridis
from bokeh.palettes import Spectral11
from bokeh.palettes import Set1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bokeh.models import ColumnDataSource, FactorRange
import numpy as np
from bokeh.palettes import Category20
from bokeh.core.properties import value
from bokeh.layouts import layout, row, column, gridplot
from bokeh.palettes import Spectral
from pymongo import MongoClient
from bokeh.io import curdoc, output_file
from html import unescape
from bokeh.io import show, save
from bokeh.models.widgets.tables import TableColumn, DataTable


# In[4]:



# I used a custom stoplist because most Nigerians informally interact in pidgin on twitter.
def stoplist():
    my_stoplist = [

"among",
"amongst",
"an",
"and",
"announce",
"another",
"any",
"anybody",
"anyhow",
"anymore",
"anyone",
"anything",
"anyway",
"anyways",
"anywhere",
"apparently",
"approximately",
"are","a",
"able",
"about",
"above",
"abst",
"accordance",
"according",
"accordingly",
"across",
"act",
"actually",
"added",
"adj",
"affected",
"affecting",
"affects",
"after",
"afterwards",
"again",
"against",
"ah",
"all",
"almost",
"alone",
"along",
"already",
"also",
"although",
"always",
"am",
"aren",
"arent",
"arise",
"around",
"as",
"aside",
"ask",
"asking",
"at",
"auth",
"available",
"away",
"awfully",
"b",
"back",
"be",
"became",
"because",
"become",
"becomes",
"becoming",
"been",
"before",
"beforehand",
"begin",
"beginning",
"beginnings",
"begins",
"behind",
"being",
"believe",
"below",
"beside",
"besides",
"between",
"beyond",
"biol",
"both",
"brief",
"briefly",
"but",
"by",
"c",
"ca",
"came",
"can",
"cannot",
"can't",
"cause",
"causes",
"certain",
"certainly",
"co",
"com",
"come",
"comes",
"contain",
"containing",
"contains",
"could",
"couldnt",
"d",
"date",
"did",
"didn't",
"different",
"do",
"does",
"doesn't",
"doing",
"done",
"don't",
"down",
"downwards",
"due",
"during",
"e",
"each",
"ed",
"edu",
"effect",
"eg",
"eight",
"eighty",
"either",
"else",
"elsewhere",
"end",
"ending",
"enough",
"especially",
"et",
"et-al",
"etc",
"even",
"ever",
"every",
"everybody",
"everyone",
"everything",
"everywhere",
"ex",
"except",
"f",
"far",
"few",
"ff",
"fifth",
"first",
"five",
"fix",
"followed",
"following",
"follows",
"for",
"former",
"formerly",
"forth",
"found",
"four",
"from",
"further",
"furthermore",
"g",
"gave",
"get",
"gets",
"getting",
"give",
"given",
"gives"
"giving",
"go",
"goes",
"gone",
"got",
"gotten",
"h",
"had",
"happens",
"hardly",
"has",
"hasn't",
"have",
"haven't",
"having",
"he",
"hed",
"hence",
"her",
"here",
"hereafter",
"hereby",
"herein",
"heres",
"hereupon",
"hers",
"herself",
"hes",
"hi",
"hid",
"him",
"himself",
"his",
"hither",
"home",
"how",
"howbeit",
"however",
"hundred",
"i",
"id",
"ie",
"if",
"i'll",
"im",
"immediate",
"immediately",
"importance",
"important",
"in",
"inc",
"indeed",
"index",
"information",
"instead",
"into",
"invention",
"inward",
"is",
"isn't",
"it",
"itd",
"it'll",
"its",
"itself",
"i've",
"j",
"just",
"k",
"keep",
"keeps",
"kept",
"kg",
"km",
"know",
"known",
"knows",
"l",
"largely",
"last",
"lately",
"later",
"latter",
"latterly",
"least",
"less",
"lest",
"let",
"lets",
"like",
"liked",
"likely",
"line",
"little",
"'ll",
"look","looking",
"looks","ltd",
"m","made",
"mainly","make",
"makes","many",
"may","maybe",
"me","mean",
"means","meantime",
"meanwhile","merely",
"mg","might",
"million","miss",
"ml","more",
"moreover""most",
"mostly","mr",
"mrs","much",
"mug","must",
"my","myself",
"n","na",
"name","namely",
"nay","nd",
"near","nearly",
"necessarily","necessary",
"need","needs",
"neither","never",
"nevertheless","new",
"next","nine",
"ninety","no",
"nobody","non",
"none","nonetheless",
"noone","nor",
"normally","nos",
"not","noted",
"nothing","now",
"nowhere","o",
"obtain","obtained",
"obviously","of",
"off","often",
"oh","ok",
"okay","old",
"omitted","on",
"once","one",
"ones","only",
"onto","or",
"ord","other",
"others","otherwise",
"ought","our",
"ours","ourselves",
"out","outside",
"over","overall",
"owing","own",
"p","page",
"pages","part",
"particular","particularly",
"past","per",
"perhaps","placed",
"please","plus",
"poorly","possible",
"possibly","potentially",
"pp","predominantly",
"present","previously",
"primarily","probably",
"promptly","proud",
"provides","put",
"q","que",
"quickly","quite",
"qv","r",
"ran","rather",
"rd","re",
"readily","really",
"recent","recently",
"ref","refs",
"regarding","regardless",
"regards",
"related","relatively",
"research","respectively",
"resulted","resulting",
"results","right",
"run","s",
"said","same",
"saw","say",
"saying","says",
"sec","section",
"see","seeing",
"seem","seemed",
"seeming","seems",
"seen","self",
"selves","sent",
"seven","several",
"shall","she",
"shed","she'll",
"shes","should",
"shouldn't","show",
"showed","shown",
"showns","shows",
"significant","significantly",
"similar","similarly",
"since","six",
"slightly","so",
"some","somebody",
"somehow","someone",
"somethan","something",
"sometime","sometimes",
"somewhat","somewhere",
"soon","sorry",
"specifically","specified",
"specify","specifying",
"still","stop",
"strongly","sub",
"substantially","successfully",
"such","sufficiently",
"suggest","sup",
"sure","t",
"take","taken",
"taking","tell",
"tends","th",
"than","thank",
"thanks","thanx",
"that","that'll",
"thats","that've",
"the","their",
"theirs","them",
"themselves","then",
"thence","there",
"thereafter""thereby"
"thered""therefore",
"therein","there'll",
"thereof","therere",
"theres","thereto",
"thereupon","there've",
"these","they",
"theyd","they'll",
"theyre","they've",
"think","this",
"those","thou",
"though","thoughh",
"thousand","throug",
"through","throughout",
"thru","thus",
"til","tip",
"to","together",
"too","took",
"toward","towards",
"tried","tries",
"truly","try",
"trying",
"ts","twice",
"two","u",
"un","under",
"unfortunately","unless",
"unlike","unlikely",
"until","unto",
"up","upon",
"ups","us",
"use","used",
"useful","usefully",
"usefulness","uses",
"using","usually",
"v","value"
"various","'ve",
"very","via",
"viz","vol",
"vols","vs",
"w","want",
"wants","was",
"wasnt","way",
"we","wed",
"welcome","we'll",
"went","were",
"werent","we've",
"what","whatever",
"what'll","whats",
"when","whence",
"whenever","where",
"whereafter","whereas",
"whereby","wherein",
"wheres","whereupon",
"wherever","whether",
"which","while",
"whim","whither",
"who","whod",
"whoever","whole","who'll",
"whom",
"whomever",
"whos",
"whose",
"why",
"widely",
"willing",
"wish",
"with","within",
"without","wont",
"words","world",
"would","wouldnt",
"www",
"xv","y",
"yes",
"yet",
"you",
"youd",
"you'll",
"your",
"youre",
"yours",
"yourself",
"yourselves",
"you've",'yrs', 
"z",'nc', 'ne', 'se',
"zero",'nw', 'aa', 'nw', 
"’s", "buhari", "muhammadu",
"president", "muhamadu", "abubakar", 
"bn", "nd", "ok", "cuz", "cos", "bcos", "his",
"has", "us", "abi", "nkor", "campaign", "presidential", 'apc', 
'pdp', 'aac', 'ypp','aap','acpn','kowa','anrp','ann','sdp', 'buhari',
'atiku', 'sowore', 'kingsley','chike','oby',  'adesina','tope','fela',"omoyele", 
'donald', "nigeria", "nigerian", "election", "vote", "buharis", "naija", "...", "yelesowore",
"thing", "man", "woman", "’'", "ur", "st'", "'️'", "ya", "ye", "–", "atk", "pls","minus", "add","'️"," ️",
"good", "bad", "hv", "gat", "…", "bý", "mk", "ng", "person", "people", "rt", 'pmp', 'pmb', 'pmbs', 'pmbs$',
"till", "until","votekingsleymoghalugetso", "voteypp", "itstime", "'s", '‘',"year",'’','•', "presidency", "retweet","aa" ]
    return my_stoplist

#Created a list of continents and countries
africa = "|".join(["Angola", 
                   "Africa",
"Burkina Faso",
"Burundi",
"Benin",
"Botswana",
"Democratic Republic of Congo",
"Central African Republic",
"Congo",
"Cote D'Ivoire",
"Cameroon",
"Cape Verde",
"Djibouti",
"Algeria",
"Egypt",
"Western Sahara",
"Eritrea",
"Ethiopia",
"Gabon",
"Ghana",
"Gambia",
"Guinea",
"Equatorial Guinea",
"Guinea-bissau",
"Kenya",
"Comoros",
"Liberia",
"Lesotho",
"Libyan Arab Jamahiriya",
"Morocco",
"Madagascar",
"Mali",
"Mauritania",
"Mauritius",
"Malawi",
"Mozambique",
"Namibia",
"Reunion",
"Rwanda",
"Seychelles",
"Sudan",
"St. Helena",
"Sierra Leone",
"Senegal",
"Somalia",
"Sao Tome and Principe",
"Swaziland",
"Chad",
"Togo",
"Tunisia",
"Tanzania, United Republic of",
"Uganda",
"Mayotte",
"South Africa",
"Zambia",
"Zimbabwe"])



asia = "|".join([
    "United Arab Emirates",
"Afghanistan",
"Armenia",
"Azerbaijan",
"Bangladesh",
"Bahrain",
"Brunei Darussalam",
"Bhutan",
"Cocos (Keeling) Islands",
"China",
"Christmas Island",
"Cyprus",
"Georgia",
"Hong Kong",
"Indonesia",
"Israel",
"India",
"British Indian Ocean Territory",
"Iraq",
"Iran (Islamic Republic of)",
"Jordan",
"Japan",
"Kyrgyzstan",
"Cambodia",
"North Korea",
"Korea, Republic of",
"Kuwait",
"Kazakhstan",
"Lao People's Democratic Republic",
"Lebanon",
"Sri Lanka",
"Myanmar",
"Mongolia",
"Macau",
"Maldives",
"Malaysia",
"Nepal",
"Oman",
"Philippines",
"Pakistan",

"Qatar",
"Russian Federation",
"Saudi Arabia",
"Singapore",
"Syrian Arab Republic",
"Thailand",
"Tajikistan",

"Turkmenistan",
"Turkey",
"Taiwan",
"Uzbekistan",
"Viet Nam",
"Yemen",
    "Syria",
    "Vietnam", "KSA"
])




europe = "|".join([
    "Andorra",
"Albania",
"Armenia",
"Austria",

"Azerbaijan",
"Bosnia and Herzegovina",
"Belgium",
"Bulgaria",
"Belarus",
"Switzerland",
"Cyprus",
"Czech Republic",
"Germany",
"Denmark",
"Estonia",
"Spain",
"Finland",
"Faroe Islands",
"France",
"United Kingdom",
"Georgia",

"Gibraltar",
"Greece",
"Croatia",
"Hungary",
"Ireland",

"Iceland",
"Italy",

"Kazakhstan",
"Liechtenstein",
"Lithuania",
"Luxembourg",
"Latvia",
"Monaco",
"Moldova, Republic of",

"Macedonia",
"Malta",
"Netherlands",
"Norway",
"Poland",
"Portugal",
"Romania",

"Russian Federation",
"Sweden",
"Slovenia",
"Svalbard and Jan Mayen Islands",
"Slovak Republic",
"San Marino",
"Turkey",
"Ukraine",
"Vatican City State (Holy See)",
    "UK",
    "Jand"
])



americas = "|".join([
    "Antigua and Barbuda",
"Anguilla",
"Netherlands Antilles",
"Aruba",
"Barbados",

"Bermuda",

"Bahamas",
"Belize",
"Canada",
"Costa Rica",
"Cuba",
"Dominica",
"Dominican Republic",
"Grenada",
"Greenland",
"Guadeloupe",
"Guatemala",
"Honduras",
"Haiti",
"Jamaica",
"Saint Kitts and Nevis",
"Cayman Islands",
"Saint Lucia",

"Martinique",
"Montserrat",
"Mexico",
"Nicaragua",
"Panama",
"St. Pierre and Miquelon",
"Puerto Rico",
"El Salvador",

"Turks and Caicos Islands",
"Trinidad and Tobago",
"United States Minor Outlying Islands",
"United States",
"Saint Vincent and the Grenadines",
"Virgin Islands (British)",
"Virgin Islands (U.S.)",
"Argentina",
"Bolivia",
"Brazil",
"Chile",
"Colombia",
"Ecuador",
"Falkland Islands (Malvinas)",
"French Guiana",
"Guyana",
"Peru",
"Paraguay",
"Suriname",
"Uruguay",
"Venezuela" ,
    "USA",
    "Yankee",
    "US",
    "America"
])




oceania = "|".join([
    
"American Samoa",
"Australia",
"Cook Islands",
"Fiji",
"Micronesia, Federated States of",
"Guam",
"Kiribati",
"Marshall Islands",
"Northern Mariana Islands",
"New Caledonia",
"Norfolk Island",
"Nauru",
"Niue",
"New Zealand",
"French Polynesia",
"Papua New Guinea",
"Pitcairn",
"Palau",
"Solomon Islands",
"Tokelau",
"Tonga",
"Tuvalu",
"Samoa"
])



#A list of the key cities in political regions in Nigeria
NC = '|'.join(["Makurdi","Benue", "Abaji","Gwagwalada","Kuje","Bwari","Kwali","abuja","Ilorin", "Kogi","Ajaokuta", "Idah","Igalamela-Odolu","Ijumu","Kabba","Lokoja","Okene", "Asa, Ilorin", "Kwara","jos", "pankshin", "Plateau", "Lafia", "Nasarawa", "Bida", "Minna", "Suleja", "Federal Capital Territory"])



NE = '|'.join(["Yola", "Jimeta", "Mubi", "Numan", "Adamawa","Azare", "bauchi", "jama'are", "Katagum", "Misau", "Bauchi", "Biu", "Dikwa", "Maiduguri", "Borno","Gombe", "Deba habe", "Kumo","Ibi", "jalingo", "Muri", "Taraba","Damaturu", "Nguru", "Yobe"])




NW = '|'.join(["Jemaa", "Kaduna", "Zaria", "Sokoto","Gusau", "Kaura Namoda", "Zamfara","Argungu", "Birnin Kebbi", "Gwandu", "Yelwa", "Kebbi","Kastina", "Daura","kano","Birnin Kudu", "Dutse", "Gumel", "Hadejia", "Kazaure", "Jigawa"])



SE = '|'.join(["Awka", "Onitsha", "Anambra", "Abakaliki", "Ebonyi", "Enugu", "Nsukka","Owerri", "Imo","Aba", "Arochukwu", "Umuahia", "Abia"])




SS = '|'.join(["Ikot Abasi", "Ikot Ekpene", "Oron", "Uyo", "Akwa Ibom", "Calabar", "ogoja", "cross river","Benin", "Edo", "Asaba", "Agbor", "Koko", "Sapele", "Ughelli", "Warri", "Delta","Bonny", "Degema", "Okrika","port harcourt", "Rivers", "Brass", "Bayelsa", "Yenagoa"])



SW = '|'.join(["Ado-Ekiti", "Ekiti","Lagos", "Ikeja", "Lekki", "Victoria Island", "Banana Island", "Ikoyi", "Eko", "lasgidi", "Abeokuta", "Shagamu", "Ogun", "Ijebu", "Akure", "Ondo", "Ikare","Oshogbo", "Osun", "Ilesha", "Ile-Ife","Oyo", "Ibadan", "Ogbomosho"  ])

my_stoplist = stoplist()

#Function to clean tweets

def clean_df(df):
    df['clean'] =  df['text'].apply(lambda x: unescape(x))
    df.clean = df.clean.apply(lambda x: x.lower())
    punct = str.maketrans(dict.fromkeys(string.punctuation))
    df.clean = df.clean.apply(lambda x: x.translate(punct))
    dig = str.maketrans(dict.fromkeys(string.digits))
    df.clean = df.clean.apply(lambda x: x.translate(dig))
    df['region'] = ""
    df.loc[df['user_location'].str.contains(NC,na=False,case = False ),'region'] = 'Nigeria - NC'
    df.loc[df['user_location'].str.contains(NE,na=False,case = False ),'region'] = 'Nigeria - NE'
    df.loc[df['user_location'].str.contains(NW,na=False,case = False ),'region'] = 'Nigeria - NW'
    df.loc[df['user_location'].str.contains(SE,na=False,case = False ),'region'] = 'Nigeria - SE'
    df.loc[df['user_location'].str.contains(SS,na=False,case = False ),'region'] = 'Nigeria - SS'
    df.loc[df['user_location'].str.contains(SW,na=False,case = False ),'region'] = 'Nigeria - SW'
    df.loc[df['user_location'].str.contains(africa,na=False,case = False ),'region'] = 'Africa'
    df.loc[df['user_location'].str.contains(americas,na=False,case = False ),'region'] = 'The Americas'
    df.loc[df['user_location'].str.contains(europe,na=False,case = False ),'region'] = 'Europe'
    df.loc[df['user_location'].str.contains(asia,na=False,case = False ),'region'] = 'Asia'
    df.loc[df['user_location'].str.contains(oceania,na=False,case = False ),'region'] = 'Oceania'
    df.loc[df['user_location'] == 'Nigeria', 'region'] = 'Other-Nigeria'
    df.loc[df['user_location'] == 'nigeria', 'region'] = 'Other-Nigeria'
    df.loc[df['user_location'] == 'NIGERIA', 'region'] = 'Other-Nigeria'
    df.loc[df['user_location'] == 'Naija', 'region'] = 'Other-Nigeria'
    df.loc[df['user_location'] == 'naija', 'region'] = 'Other-Nigeria'
    df.loc[df['user_location'] == 'NAIJA', 'region'] = 'Other-Nigeria'
    df.region.replace("", 'Others', inplace = True)
    df['created'] = pd.to_datetime(df['created'])
    df =  df.set_index('created')
    return df



 #Function to process tweets
def process_texts(docs):
    #import spacy
    #from spacy.lang.en.stop_words import STOP_WORDS
    #stopword = STOP_WORDS
    
    punct = string.punctuation
    nlp = spacy.load('en_core_web_sm')
    texts=[]
    counter =1
    for doc in docs:
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in my_stoplist and tok not in punct ]
        tokens ='  '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)




def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words



def check_word_in_data(word, df):
    """Checks if a word is in a Twitter dataset's text. 
    Checks text and extended tweet (140+ character tweets) for tweets,
    retweets and quoted tweets.
    Returns a logical pandas Series.
    """
    contains_column = df['clean'].str.contains(word, case = False)
    return contains_column





#Function to create a data subset for a given region
def region_text(region,df):
    newdf = df[(df.region == region)]
    return newdf[['clean','label']]




#Function to create a subset of all negative tweets
def label_neg(word, df):
    new =  df[check_word_in_data(word, df)]
    neg = [text for text in new[new['label'] == 'Negative']['clean']]
    return neg
    

#Function to create a subset of all positive tweets

def label_pos(word, df):
    new =  df[check_word_in_data(word, df)]
    pos = [text for text in new[new['label'] == 'Positive']['clean']]
    return pos
	
	
 #Function to create a subset of negative tweets for a given candidate
def neg_clean_candidate(word, df):
    docs = label_neg(word, df)
    words = process_texts(docs)
    texts = remove_non_ascii(words)
    texts = ' '.join(pd.Series(texts)).split()
    filtered_text_neg = [word for word in texts if word not in my_stoplist ]
    neg_counts = Counter(filtered_text_neg)
    return filtered_text_neg, neg_counts


#Function to create a subset of positive tweets for a given candidate

def pos_clean_candidate(word, df):

    docs = label_pos(word, df)
    words = process_texts(docs)
    texts = remove_non_ascii(words)
    texts = ' '.join(pd.Series(texts)).split()
    filtered_text_pos = [word for word in texts if word not in my_stoplist ]
    pos_counts = Counter(filtered_text_pos)
    return filtered_text_pos, pos_counts





    


#Function to return a count of tweets for a given name
def count_label(name, label):
    new =  df[check_word_in_data(name, df)]
    chec = Counter([text for text in new[new['label'] == label]['label']]).most_common()
    check = [word[1] for word in chec]
    return int(pd.Series(check))


def label_list(list_name, label):
    labl =[]
    for name in lower(list_name):
          return labl.append(count_label(name, label))
        
        
#Function to return daily sentiment for a given name (candidate or party)        
def check_sent_daily(name):
    name = name
    sent = sentiment[check_word_in_data(name, df)].resample('D').mean()
    return sent


#Function to return weekly sentiment for a given name (candidate or party)        

def check_sent_week(name):
    name = name
    sent = sentiment[check_word_in_data(name, df)].resample('W').mean()
    return sent


# In[5]:




#function to plot bar charts of mentions
def pie_chart(data, column_name, column_num, name):
    from bokeh.plotting import figure
    p = figure(y_range =data[column_name], plot_width=400, plot_height=400, title = 'Distribution of Mentions - ' + name)
    p.hbar(y=data[column_name], height=0.5, left=0,right=data[column_num], color=data['color'])
    #tab = Panel(child =p, title ='Mentions -' + name)
    return p

#Fucntion to create data of mentions and to instantiate the plot function
def plot_mention(df):
    pie_data = {
    'candidates' : ['Buhari', 'Atiku', 'Sowore', 'Moghalu'],
    'candidates_sum' :[np.sum(check_word_in_data('buhari', df)),
                       np.sum(check_word_in_data('atiku', df)),
                       np.sum(check_word_in_data('sowore', df)),
                       np.sum(check_word_in_data('kingsley|moghalu', df)),
                       ],
    
    'color' : Spectral[4],








    'parties' : ['APC', 'PDP', 'AAC', 'YPP'], 
    'parties_sum' : [np.sum(check_word_in_data('apc', df)),
                 np.sum(check_word_in_data('pdp', df)),
                 np.sum(check_word_in_data('aac', df)),
                 np.sum(check_word_in_data('ypp', df)),]
        }
    plot1 = pie_chart(data = pie_data, column_name = 'candidates', column_num ='candidates_sum', name = 'Candidates' )

    plot2 = pie_chart(data = pie_data, column_name = 'parties', column_num ='parties_sum', name = 'Parties' )
    
    #l = layout([[plot1],[plot1]])
    #tab = Panel(child =l, title ='Distribution of Mentions')
    tabb = row([plot1, plot2])
    #tab1 = Panel(child = plot1, title = 'MEntions Cand')
    #p = Panel(child = tabb, title = 'Mentions Parties')
    #tab = Tabs(tabs = [tab2])
    return tabb


# In[6]:


#Function to plot proportion of tweets
def visual_prop(df,name):
    from bokeh.plotting import figure
    from bokeh.models import NumeralTickFormatter
    pd.options.display.float_format = '{:.2%}'.format
    length = len(df.columns)
    data = {'xs' : [df.index] * length,
             'ys' : [df[columns].values for columns in df],
             'labels' : [columns for columns in df],
             'sum_palletes' : Spectral11[0:length]
    }


    source = ColumnDataSource(data)
    
    plot = figure(width=600, height=500, x_axis_type="datetime", title = "Proportional Distribution of Mentions for "+ name)
    plot.xaxis.axis_label ='Date'
    plot.yaxis.axis_label = 'Proportion of  Daily Tweets'
    plot.xaxis.major_label_orientation = 0.6
    plot.xaxis[0].formatter.days= '%m/%d/%Y'
    plot.yaxis.formatter = NumeralTickFormatter(format='0 %')
    plot.multi_line(xs='xs', ys='ys', legend='labels',line_width=3, line_color = 'sum_palletes', source=source)
    return plot





#Function to plot sum of tweets
def visual_sum(df,name):
    from bokeh.plotting import figure
    length = len(df.columns)
    data = {'xs' : [df.index] * length,
             'ys' : [df[columns].values for columns in df],
             'labels' : [columns for columns in df],
             'sum_palletes' : Spectral11[0:length]
    }

    source = ColumnDataSource(data)
    plot = figure(width=600, height=500, x_axis_type="datetime", title = "Sum of Daily Mentions for "+ name)
    plot.xaxis.axis_label ='Date'
    plot.yaxis.axis_label = 'Sum of  Daily Mentions'
    plot.xaxis.major_label_orientation = 0.6
    plot.xaxis[0].formatter.days= '%m/%d/%Y'
    plot.multi_line(xs='xs', ys='ys', legend='labels',line_width=3, line_color = 'sum_palletes', source=source)
    
    return plot






#Function to create the grid of trend (proportion, sum, cumsum, daily sum, daily cumsum)  of mentions
def draw_line_mentions(df):
    candi_daily_prop = pd.DataFrame({
	"Buhari" : check_word_in_data('buhari', df).resample('D').mean(),
	"Atiku" : check_word_in_data('atiku', df).resample('D').mean(),
	"Sowore" : check_word_in_data('sowore', df).resample('D').mean(),
	"kingsley" :  check_word_in_data('kingsley|moghalu', df).resample('D').mean()})








    daily_sum_candidate = pd.DataFrame({
	'Buhari' : check_word_in_data('buhari', df).resample('D').sum(),
	"Atiku" : check_word_in_data('atiku', df).resample('D').sum(),
	"Sowore" : check_word_in_data('sowore', df).resample('D').sum(),
	"Moghalu" : check_word_in_data('kingsley|moghalu', df).resample('D').sum()
		})






    candi_daily_cumsum = pd.DataFrame({
	'Buhari' : daily_sum_candidate.Buhari.cumsum(),
	"Atiku" : daily_sum_candidate.Atiku.cumsum(),
	"Sowore" : daily_sum_candidate.Sowore.cumsum(),
	"Moghalu" : daily_sum_candidate.Moghalu.cumsum(),})






    daily_sum_party = pd.DataFrame({
		"APC" : check_word_in_data('apc', df).resample('D').sum(),
		"PDP" : check_word_in_data('pdp', df).resample('D').sum(),
		"AAC" : check_word_in_data('aac', df).resample('D').sum(),
		"YPP" : check_word_in_data('ypp', df).resample('D').sum(),})






    party_daily_cumsum = pd.DataFrame({
	"APC" : daily_sum_party.APC.cumsum(),
	"PDP" : daily_sum_party.PDP.cumsum(),
	"AAC" : daily_sum_party.AAC.cumsum(),
	"YPP" :daily_sum_party.YPP.cumsum()})





    party_prop = pd.DataFrame({
 	'apc' : check_word_in_data('apc', df).resample('D').mean(),
  	'pdp' : check_word_in_data('pdp', df).resample('D').mean(),
  	'aac' :check_word_in_data('aac', df).resample('D').mean(),
  	'ypp' : check_word_in_data('ypp', df).resample('D').mean()
	})

    cand_daily_prop = visual_prop(candi_daily_prop,'Candidates')
    candi_daily_sum = visual_sum(daily_sum_candidate, "Each Candidate")
    candi_daily_cumsum = visual_sum(candi_daily_cumsum, "Candidates (Cumulative)")
    party_daily_sum = visual_sum(daily_sum_party, "Each Party")
    party_daily_cumsum = visual_sum(party_daily_cumsum, "Parties(Cumulative)")
    party_daily_prop = visual_prop(party_prop,'Candidates')

    #l = layout([[cand_daily_prop,candi_daily_sum, candi_daily_cumsum], [party_daily_prop,party_daily_sum, party_daily_cumsum]])
    #tab = Panel(child =l, title ='Daily Mentions')

    grid = gridplot([[cand_daily_prop,candi_daily_sum,candi_daily_cumsum ], [party_daily_sum,party_daily_cumsum, party_daily_prop]])
    #pan = Panel(child = grid, title = 'Avg Daily Mentions')
    #tab = Tabs(tabs = [pan])
    return grid




# In[7]:



#Function to return a count of mentions per region
def count_region(df,name, label, region):
    new =  df[check_word_in_data(name, df)]
    label_df = new[new['label'] == label]
    reg = Counter([place for place in label_df[label_df['region'] == region]['region']]).most_common()
    regi = [value[1] for value in reg]

    
    return regi



def clean_region(lst):
        result = [x or [0] for x in lst]
        result = list(np.array(result).flat)
        return result





#Function to create the visualization object for sentiment distribution per region per candidate or party
def visual_region_group(source, regions, factors, word):
    p = figure(x_range=FactorRange(*factors), plot_height=400, plot_width = 950, title = 'Distribution of Sentiments by Region - ' + word )
    p.vbar_stack(regions, x='x', width=0.9, alpha=0.5, color=Category20[13], source=source,
             legend=[value(x) for x in regions])
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_right"
    p.legend.orientation = "horizontal"
    
    return p






#Takes list of candidates or region and creates a plot object of the sentiment distribution
def create_plot_region(df,cand_or_party, word):
	regions= ["NC","NE","NW","SE","SS","SW","Africa","Americas","Europe","Asia","Oceania", "Unclassified", "Other_Nigeria"]
	label = ['Positive', 'Neutral', 'Negative']
	factors = [(name,typ) for name in cand_or_party for typ in label]
	NC = clean_region([count_region(df,name, lab, 'Nigeria - NC') for name in [x.lower() for x in cand_or_party] for lab in label])
	NE = clean_region([count_region(df,name, lab, 'Nigeria - NE') for name in [x.lower() for x in cand_or_party] for lab in label])
	NW = clean_region([count_region(df,name, lab, 'Nigeria - NW') for name in [x.lower() for x in cand_or_party] for lab in label])
	SE = clean_region([count_region(df,name, lab, 'Nigeria - SE') for name in [x.lower() for x in cand_or_party] for lab in label])
	SS = clean_region([count_region(df,name, lab, 'Nigeria - SS') for name in [x.lower() for x in cand_or_party] for lab in label])
	SW = clean_region([count_region(df,name, lab, 'Nigeria - SW') for name in [x.lower() for x in cand_or_party] for lab in label])
	africa = clean_region([count_region(df,name, lab, 'Africa') for name in [x.lower() for x in cand_or_party] for lab in label])
	Americas = clean_region([count_region(df,name, lab, 'The Americas') for name in [x.lower() for x in cand_or_party] for lab in label])
	Europe = clean_region([count_region(df,name, lab, 'Europe') for name in [x.lower() for x in cand_or_party] for lab in label])
	Asia = clean_region([count_region(df,name, lab, 'Asia') for name in [x.lower() for x in cand_or_party] for lab in label])
	Oceania = clean_region([count_region(df,name, lab, 'Oceania') for name in [x.lower() for x in cand_or_party] for lab in label])
	Mislabelled = clean_region([count_region(df,name, lab, 'Others') for name in [x.lower() for x in cand_or_party] for lab in label])
	Other_Nigeria = clean_region([count_region(df,name, lab, 'Nigeria') for name in [x.lower() for x in cand_or_party] for lab in label])








	source = ColumnDataSource(data=dict(x = factors, 
				    NC = NC,
                                    NE = NE, 
                                    NW = NW,
                                    SE = SE,
                                    SS = SS,
                                    SW = SW,
                                    Africa = africa,
                                    Americas= Americas,
                                    Europe = Europe,
                                    Asia = Asia,
                                    Oceania = Oceania,
                                    Unclassified = Mislabelled,
                                   Other_Nigeria = Other_Nigeria))




	map_data = visual_region_group(source, regions, factors, word)

	return map_data




#Returns the tabs for regional sentiment distribution per candidates and parties
def return_plot_region(df):
	#plot candidates 
    candidates =  ['Buhari', 'Atiku', 'Sowore', 'Moghalu']
    cand = create_plot_region(df,candidates, "Candidates")
	#plot parties
    parties =  ['APC', 'PDP', 'AAC', 'YPP']
    party = create_plot_region(df,parties, "Parties")
    l = gridplot([[cand],[party]])
    #pan = Panel(child = l, title= 'sentiment by regions')
    #tab = Tabs(tabs = [pan])
    return l


# In[8]:



#Function to return plot of trend of average sentiment per candidate or party
def visual_sentiment(df, name):
    from bokeh.plotting import figure

    length = len(df.columns)
    data = {'xs' : [df.index] * length,
             'ys' : [df[columns].values for columns in df],
             'labels' : [columns for columns in df],
             'sum_palletes' : Spectral11[0:length]
    }

    source = ColumnDataSource(data)
    plot = figure(y_range=(-1,1), width=600, height=500, x_axis_type="datetime", title = 'Average Sentiments - ' + name )
    plot.xaxis.axis_label ='Date'
    plot.yaxis.axis_label = 'Average Sentiment'
    plot.xaxis.major_label_orientation = 0.6
    plot.xaxis[0].formatter.days= '%m/%d/%Y'
    plot.multi_line(xs='xs', ys='ys', legend='labels',line_width=3, line_color = 'sum_palletes', source=source)
    return plot



#Function to return count of sentiments
def label_plot(cand_or_party, label_data, name):
    from bokeh.models import ColumnDataSource, FactorRange
    from bokeh.transform import factor_cmap
    labels = ['Negative', 'Neutral', 'Positive']
    x = [(candidate, label) for candidate in cand_or_party for label in labels  ]
    counts = sum(zip(label_data['Positive'], label_data['Neutral'], label_data['Negative']), ())
    source = ColumnDataSource(data = dict(x=x, counts = counts))
    p = figure(x_range=FactorRange(*x), plot_height=250, title="Sentiment Count By "+ name)
    p.vbar(x='x', top='counts', width=0.9, source=source, line_color = 'white',
      fill_color=factor_cmap('x', palette=label_data['color'], factors=labels, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    return p



#Function to return count of sentiment per given candidate or party
def count_label(name, label, df):
    new =  df[check_word_in_data(name, df)]
    chec = Counter([text for text in new[new[label] == label][label]]).most_common(1)
    check = [word[1] for word in chec]
    return int(pd.Series(check))


        
        
 #Function to return daily sentiment for a given candisate or party       
def check_sent_daily(name,sentiment,df):
    name = name
    sent = sentiment[check_word_in_data(name, df)].resample('D').mean()
    return sent


 #Function to return weekly sentiment for a given candisate or party       

def check_sent_week(name,sentiment,df ):
    name = name
    sent = sentiment[check_word_in_data(name, df)].resample('W').mean()
    return sent






#Function to return grid object of sentiment count plot
def plot_label_count(df):
    candidates =  ['Buhari', 'Atiku', 'Sowore', 'Moghalu']
    parties = ['APC', 'PDP', 'AAC', 'YPP']
    label_data_cand = {
    'candidates' : candidates,
    'Positive'  : [count_label(name, 'Positive', df) for name in [x.lower() for x in candidates]],
    'Neutral'  : [count_label(name, 'Neutral', df) for name in [x.lower() for x in candidates]],
    'Negative' :[count_label(name, 'Negative', df) for name in [x.lower() for x in candidates]],  
  	'color' : Set1[3]}




    label_data_party = {
	'parties' : parties, 
	'Positive' : [count_label(name, 'Positive', df) for name in [x.lower() for x in parties]],
	'Neutral' :[count_label(name, 'Neutral', df) for name in [x.lower() for x in parties]],
	'Negative' :[count_label(name, 'Negative', df) for name in [x.lower() for x in parties]],
	'color' : Set1[3]
	}	

    candi_plot = label_plot(candidates, label_data_cand, 'Candidates')
    party_plot = label_plot(parties, label_data_party, 'Parties')
    l = gridplot([[candi_plot], [party_plot]], plot_width = 550)
    #pan = Panel(child =l, title ='Sentiment Count')
    #tab = Tabs(tabs = [pan])
    return l







#Function to return grid object of weekly avg sentiment plot
def plot_sentiment_avg_daily_weekly(df):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = df.clean.apply(sid.polarity_scores)
    sentiment = sentiment_scores.apply(lambda x: x['compound'])
    daily_sent_candi = pd.DataFrame({
    "Buhari" : check_sent_daily('buhari', sentiment, df),
    "Atiku" : check_sent_daily('atiku', sentiment, df),
    "Sowore" : check_sent_daily('sowore', sentiment, df),
    "Moghalu" : check_sent_daily('kingsley|moghalu', sentiment, df)
    
	})



    daily_sent_party = pd.DataFrame({
    
    
	"APC" : check_sent_daily('apc', sentiment, df),
	"PDP" : check_sent_daily('pdp', sentiment, df),
	"AAC" : check_sent_daily('aac', sentiment, df),
	"YPP" : check_sent_daily('ypp', sentiment, df)
	})
    
    



    weekly_sent_candi = pd.DataFrame({

	"Buhari" : check_sent_week('buhari', sentiment, df),
	"Atiku" : check_sent_week('atiku', sentiment, df),
	"Sowore" : check_sent_week('sowore', sentiment, df),
	"Moghalu": check_sent_week('kingsley|moghalu', sentiment, df)
	})

#proportion Party

    weekly_sent_party = pd.DataFrame({


	"APC" : check_sent_week('apc', sentiment, df),
	"PDP" : check_sent_week('pdp', sentiment, df),
	"AAC" : check_sent_week('aac', sentiment, df),
	"YPP" : check_sent_week('ypp', sentiment, df)
	})



    daily_cand = visual_sentiment(daily_sent_candi,'Candidates (Daily)')

    daily_party =visual_sentiment(daily_sent_party,'Parties (Daily)')



    weekly_candy = visual_sentiment(weekly_sent_candi,'Candidates (Weekly)')

    weekly_party = visual_sentiment(weekly_sent_party,'Parties (Weekly)')
    l = gridplot([[daily_cand,daily_party ],[weekly_candy,weekly_party]], plot_width = 550)
    #p = Panel(child =l, title ='Average Sentiments')
    #tab = Tabs(tabs =[p])
    return l 




# In[9]:

def process_texts(docs):
    #import spacy
    #from spacy.lang.en.stop_words import STOP_WORDS
    #stopword = STOP_WORDS
    
    punct = string.punctuation
    nlp = spacy.load('en_core_web_sm')
    texts=[]
    counter =1
    for doc in docs:
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in my_stoplist and tok not in punct ]
        tokens ='  '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)



#Function to return word count for negative terms of interest

def neg_terms_of_interest(word, df):
    stop = stopwords.words('english')
    docs = label_neg(word, df)
    words = process_texts(docs)
    texts = remove_non_ascii(words)
    texts = ' '.join(pd.Series(texts)).split()
    filtered_text_neg = [word for word in texts if word not in stop ]
    neg_counts = Counter(filtered_text_neg)
    return filtered_text_neg, neg_counts


#Function to return plot objects of negative terms of interest.

def visualize_interest(neg_word, neg_count,name):
    type_neg = 'Negative'
    plot_1=plot_top_common_words(word_counts =neg_count , name =name, typeof = type_neg )
    #plot_2=draw_cloud(neg_word, name, type_neg )
    plot_3=plot_30_common_bigrams(word = neg_word, name = name, typeof =type_neg )
    #plot_4=plot_30_common_trigrams(word = neg_word, name = name, typeof =type_neg)
    neg =  [plot_1, plot_3]
    
    
    
    return neg




#Instantiation of negative terms of interest
def plot_special_interest(df):
    corruption_neg_words,corruption_neg_count = neg_terms_of_interest('corruption|corrupt|embezzle|Steal|loot|looters|fraud|siphon', df)
    plot1 = visualize_interest(corruption_neg_words, corruption_neg_count, 'Corruption')

    underage_neg_words,underage_neg_count = neg_terms_of_interest('underage voters|underage| under age| rig election| buy votes| buying votes', df)
    plot2 = visualize_interest(underage_neg_words, underage_neg_count, 'Underage Voters')

    voters_neg_words,voters_neg_count = neg_terms_of_interest('voter|vote|voting', df)
    plot3 =visualize_interest(voters_neg_words, voters_neg_count,  'Voters')

    electricity_neg_words,electricity_neg_count = neg_terms_of_interest('electricity|nepa|phcn|light', df)
    plot4 =visualize_interest(electricity_neg_words, electricity_neg_count,  'Electricity, NEPA & PHCN')

	# ### Distribution of Tweets mentioning Key Ethnic Groups
    igbo_neg_words,igbo_neg_count = neg_terms_of_interest('igbo|ibo', df)
    plot5 =visualize_interest(igbo_neg_words, igbo_neg_count,  'the Igbo Ethnic Group')

    hausa_neg_words,hausa_neg_count = neg_terms_of_interest('hausa', df)
    plot6 =visualize_interest(hausa_neg_words, hausa_neg_count,  'the Hausa Ethnic Group')

    yoruba_neg_words,yoruba_neg_count = neg_terms_of_interest('yoruba', df)
    plot7 = visualize_interest(yoruba_neg_words, yoruba_neg_count, 'the Yoruba Ethnic Group')

    l = gridplot([plot1, plot2, plot3, plot4, plot5, plot6, plot7], plot_width = 850)
    #pan = Panel(child = l, title = "Negative - Special Interest")
    #tab = Tabs(tabs = [pan])
    return l


# In[10]:




#Instantiation of stopwords
my_stoplist = stoplist()

#Remove non-ascii words.  
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


    
    
    
    
    
    
def process_texts(docs):
    import spacy
    #from spacy.lang.en.stop_words import STOP_WORDS
    #stopword = STOP_WORDS
    
    punct = string.punctuation
    nlp = spacy.load('en_core_web_sm')
    texts=[]
    counter =1
    for doc in docs:
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in my_stoplist and tok not in punct ]
        tokens ='  '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)







    
    
    
#FUnction to return count and words in negative tweets    
def neg_clean_candidate(word, df):
    docs = label_neg(word, df)
    words = process_texts(docs)
    texts = remove_non_ascii(words)
    texts = ' '.join(pd.Series(texts)).split()
    filtered_text_neg = [word for word in texts if word not in my_stoplist ]
    neg_counts = Counter(filtered_text_neg)
    return filtered_text_neg, neg_counts



#FUnction to return count and words in positive tweets
def pos_clean_candidate(word, df):

    docs = label_pos(word, df)
    words = process_texts(docs)
    texts = remove_non_ascii(words)
    texts = ' '.join(pd.Series(texts)).split()
    filtered_text_pos = [word for word in texts if word not in my_stoplist ]
    pos_counts = Counter(filtered_text_pos)
    return filtered_text_pos, pos_counts






#def draw_cloud(data, name, typeof):
#    name = name
#    cloud = WordCloud(max_words = 30000, max_font_size =50, background_color="black").generate(str(data))
#    plt.figure(figsize =(10,8), facecolor = 'k')
#    plt.imshow(cloud, interpolation ='bilinear')
#    plt.title(typeof + 'Words - ' + name)
#    plt.axis('off')
 #   plt.tight_layout(pad=0)
 #   plt.show()
    
    
    
    

    
    
    
    
    



#Function to plot top 30 words
def plot_top_common_words(word_counts, name, typeof):
    common_words = [word[0] for word in word_counts.most_common(30)]
    common_counts = [word[1] for word in word_counts.most_common(30)]
    
    bar = figure(y_range = common_words, plot_width=850, plot_height=400, title = 'Distribution of Top 30 Common Words in -'+ typeof +' Tweets - ' + name)
    bar.hbar(y =common_words, height=0.5, left=0, right = common_counts,  color = viridis(30) )
    bar.xgrid.grid_line_color = None
    bar.xaxis.major_label_orientation = 0.6
    
    return bar





#Function to plot top 30 bigrams
def plot_30_common_bigrams(word, name, typeof):
    words_2g = ngrams((word), 2)
    words2g_list = [' '.join(grams) for grams in words_2g]
    words_2g_count = Counter(words2g_list)
    common_words = [word[0] for word in words_2g_count.most_common(30)]
    common_counts = [word[1] for word in words_2g_count.most_common(30)]


    bar = figure(y_range = common_words, plot_width=850, plot_height=400, title = 'Distribution of Top 30 Bigrams in -'+ typeof +' Tweets - ' + name)
    bar.hbar(y =common_words,height=0.5, left=0, right = common_counts,  color = viridis(30) )
    bar.xgrid.grid_line_color = None
    bar.xaxis.major_label_orientation = 0.6
    
    return bar


#Function to plot top 30 trigrams
def plot_30_common_trigrams(word, name, typeof):
    words_3g = ngrams((word), 3)
    words3g_list = [' '.join(grams) for grams in words_3g]
    words_3g_count = Counter(words3g_list)

    common_words = [word[0] for word in words_3g_count.most_common(30)]
    common_counts = [word[1] for word in words_3g_count.most_common(30)]


    bar = figure(y_range = common_words, plot_width=850, plot_height=400, title = 'Distribution of Top 30 Trigrams in -' + typeof +  'Tweets - ' + name)
    bar.hbar(y =common_words, height=0.5, left=0, right = common_counts, color = viridis(30) )
    bar.xgrid.grid_line_color = None
    bar.xaxis.major_label_orientation = 0.6
   
    return bar




#Returns tweets and sentiment label for a given region
def region_text(region,df):
    newdf = df[(df.region == region)]
    return newdf[['clean','label']]


#Visualize common words and bigrams
def visualize(neg_word, neg_count, pos_word, pos_count, name, type_pos, type_neg):
    #from bokeh.layouts import  gridplot
    plot_1=plot_top_common_words(word_counts =neg_count , name =name, typeof = type_neg )
    #plot_2=draw_cloud(neg_word, name, type_neg )
    plot_3=plot_30_common_bigrams(word = neg_word, name = name, typeof =type_neg )
    #plot_4=plot_30_common_trigrams(word = neg_word, name = name, typeof =type_neg)
    
    
    plot_5=plot_top_common_words(word_counts =pos_count , name =name, typeof = type_pos)
    #plot_6=draw_cloud(pos_word, name,type_pos )
    plot_7=plot_30_common_bigrams(word = pos_word, name = name, typeof =type_pos)
    #plot_8=plot_30_common_trigrams(word = pos_word, name = name, typeof =type_pos)
    #col = gridplot([[plot_1,plot_3], [plot_5, plot_7] ])
    #pan = Panel(child = col, title = "Word Distribution  - " + name)
    #tab = Tabs(tabs = [pan])
    return plot_1,plot_3,plot_5, plot_7



#Visualize negative and positive sentiments for a given candidate/party for a given region
def visualize_region(word, df, type_pos, type_neg, region):
    neg_word, neg_count = neg_clean_candidate(word, df)
    pos_words, pos_count = pos_clean_candidate(word, df)
    name = word.capitalize() + "" + region
    return visualize(neg_word, neg_count, pos_words, pos_count, name, type_pos, type_neg)









# In[11]:














def draw_word_distribution(df):
    #negative and positive all candidates
    neg_word_buhari, neg_buhari_count = neg_clean_candidate('buhari', df)

    atiku_neg_words, atiku_neg_count = neg_clean_candidate('atiku', df)

    sowore_neg_words, sowore_neg_count = neg_clean_candidate('sowore', df)

    kingsley_neg_words, kingsley_neg_count = neg_clean_candidate('kingsley|moghalu', df)

    pos_word_buhari, pos_buhari_count = pos_clean_candidate('buhari', df)

    atiku_pos_words, atiku_pos_count = pos_clean_candidate('atiku', df)

    sowore_pos_words, sowore_pos_count = pos_clean_candidate('sowore', df)

    kingsley_pos_words, kingsley_pos_count = pos_clean_candidate('kingsley|moghalu', df)
   
    #negative and positive all parties

    apc_neg_words, apc_neg_count = neg_clean_candidate('apc', df)

    pdp_neg_words, pdp_neg_count = neg_clean_candidate('pdp', df)

    aac_neg_words, aac_neg_count = neg_clean_candidate('aac', df)

    ypp_neg_words, ypp_neg_count = neg_clean_candidate('ypp', df)

    apc_pos_words, apc_pos_count = pos_clean_candidate('apc', df)

    pdp_pos_words, pdp_pos_count = pos_clean_candidate('pdp', df)

    aac_pos_words, aac_pos_count = pos_clean_candidate('aac', df)

    ypp_pos_words, ypp_pos_count = pos_clean_candidate('ypp', df)





    plot1,plot2,plot3,plot4 = visualize(neg_word_buhari, neg_buhari_count, pos_word_buhari, pos_buhari_count, 'Buhari', 'Positive', 'Negative')
    plot5, plot6,plot7,plot8 = visualize(atiku_neg_words, atiku_neg_count, atiku_pos_words, atiku_pos_count, 'Atiku', 'Positive', 'Negative')
    plot9,plot10,plot11,plot12 = visualize(sowore_neg_words, sowore_neg_count, sowore_pos_words, sowore_pos_count, 'Sowore', 'Positive', 'Negative')
    plot13,plot14,plot15,plot16 = visualize(kingsley_neg_words, kingsley_neg_count, kingsley_pos_words, kingsley_pos_count, 'Moghalu', 'Positive', 'Negative')


    # ## Parties
    plot17, plot18, plot19,plot20 = visualize(apc_neg_words, apc_neg_count, apc_pos_words, apc_pos_count, 'APC', 'Positive', 'Negative')
    plot21,plot22,plot23,plot24 = visualize(pdp_neg_words, pdp_neg_count, pdp_pos_words, pdp_pos_count, 'PDP', 'Positive', 'Negative')
    plot25,plot26,plot27,plot28 = visualize(aac_neg_words, aac_neg_count, aac_pos_words, aac_pos_count, 'AAC', 'Positive', 'Negative')
    plot29,plot30,plot31,plot32 = visualize(ypp_neg_words, ypp_neg_count, ypp_pos_words, ypp_pos_count, 'YPP','Positive', 'Negative')
    col = gridplot([[plot1,plot2,plot3,plot4],[plot5,plot6,plot7,plot8],[plot9,plot10,plot11,plot12],[plot13,plot14,plot15,plot16],[plot17,plot18,plot19,plot20],[plot21,plot22,plot23,plot24],[plot25,plot26,plot27,plot28],[plot29,plot30,plot31,plot32] ], plot_width = 550)
    #pan = Panel(child = col, title = "Unique Word Distribution")


    return col


# In[ ]:




# In[ ]:




# In[ ]:



# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:





#return visualization grids for regions
def visualize_region(word, df, type_pos, type_neg, region):
    neg_word, neg_count = neg_clean_candidate(word, df)
    pos_words, pos_count = pos_clean_candidate(word, df)
    name = word.capitalize() + "" + region
    plot1, plot2, plot3, plot = visualize(neg_word=neg_word, neg_count=neg_count, pos_word=pos_words, pos_count=pos_count, name=name, type_pos = type_pos, type_neg = type_neg)
    return plot1, plot2, plot3, plot












#Return panel of regions
def draw_word_distribution_regions(df, name):
	 


    word = name.lower()
    NC_df = region_text('Nigeria - NC',df)
    NE_df = region_text('Nigeria - NE',df)
    NW_df = region_text('Nigeria - NW',df)
    SE_df = region_text('Nigeria - SE',df)
    SS_df = region_text('Nigeria - SS',df)
    SW_df = region_text('Nigeria - SW',df)
    Africa_df = region_text('Africa',df)
    Americas_df = region_text('The Americas',df)
    Europe_df = region_text('Europe',df)
    Asia_df = region_text('Asia',df)
    Oceania_df = region_text('Oceania',df)
    Other_Nig_df = region_text('Other-Nigeria',df)
    No_label_df = region_text('Others',df)




    plot1, plot2, plot3, plot4  = visualize_region(word, NC_df, 'Positive', 'Negative', "(North Central Nigeria)")

    plot5, plot6, plot7, plot8 = visualize_region(word, NE_df, 'Positive', 'Negative', "(North East Nigeria)")

    plot9, plot10, plot11, plot12 = visualize_region(word, NW_df, 'Positive', 'Negative', "(North West Nigeria)")

    plot13, plot14, plot15, plot16 = visualize_region(word, SE_df, 'Positive', 'Negative', "(South East Nigeria)")

    plot17, plot18, plot19, plot20 = visualize_region(word, SS_df, 'Positive', 'Negative', "(South South Nigeria)")

    plot21, plot22, plot23, plot24 = visualize_region(word, SW_df, 'Positive', 'Negative', "(South West Nigeria)")


    plot25, plot26, plot27, plot28 =visualize_region(word, Other_Nig_df, 'Positive', 'Negative', "(Other Nigeria)")


    plot29, plot30, plot31, plot32 = visualize_region(word, Africa_df, 'Positive', 'Negative', "(Africa)")

    plot33, plot34, plot35, plot36 = visualize_region(word, Americas_df, 'Positive', 'Negative', "(The Americas)")

    plot37, plot38, plot39, plot40 = visualize_region(word, Europe_df, 'Positive', 'Negative', "(Europe)")

    plot41, plot42, plot43, plot44 =visualize_region(word, Asia_df, 'Positive', 'Negative', "(Asia)")

    plot45, plot46, plot47, plot48 = visualize_region(word, Oceania_df, 'Positive', 'Negative', "(Oceania)")

    plot49, plot50, plot51, plot52 = visualize_region(word, No_label_df, 'Positive', 'Negative', "(Unclassified)")


    col = gridplot([[plot1,plot2,plot3,plot4],[plot5,plot6,plot7,plot8],[plot9,plot10,plot11,plot12],[plot13,plot14,plot15,plot16],[plot17,plot18,plot19,plot20],[plot21,plot22,plot23,plot24],[plot25,plot26,plot27,plot28],[plot29,plot30,plot31,plot32],[plot33,plot34,plot35,plot36],[plot37,plot38,plot39,plot40],[plot41,plot42,plot43,plot44],[plot45,plot46,plot47,plot48],[plot49,plot50,plot51,plot52] ], plot_width = 550)

    pan = Panel(child = col, title = name + "(regions)")
    #tab = Tabs(tabs = [pan])
    return pan



# In[ ]:




# In[ ]:




# In[ ]:




# In[13]:
#Connect to MongoDB
client = MongoClient('mongodb://gilbert:Akanchawa123$@cluster0-shard-00-00-xkbsr.gcp.mongodb.net:27017,cluster0-shard-00-01-xkbsr.gcp.mongodb.net:27017,cluster0-shard-00-02-xkbsr.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true')

#COnnect to DB
db = client['politics']

#Connect to Collection
collection = db['fordb']



#Extract Collection into Dataframe
data_list = list(collection.find())
data = pd.DataFrame(data_list)



# In[15]:
#Instatiate data cleaning function
df = clean_df(data)


# In[16]:
#Instantiate mention plots
tab1 = plot_mention(df)
p1 = Panel(child = tab1, title = 'Mentions Parties')

#Instantiate mention trend plot
tab2 = draw_line_mentions(df)
p2 = Panel(child = tab2, title = 'Avg Daily Mentions')


# In[ ]:


#Istantiate region plots
tab3 = return_plot_region(df)
p3 = Panel(child = tab3, title= 'Sentiment by Region')


# In[ ]:




# In[ ]:
#Instantiate sentiment count
tab4 = plot_label_count(df)
p4 = Panel(child =tab4, title ='Sentiment Count')




# In[ ]:

#Istantiate sentiment trend plot
tab5 = plot_sentiment_avg_daily_weekly(df)
p5 = Panel(child =tab5, title ='Average Sentiments (Daily & Weekly)')


# In[ ]:


#Instantiate plot of special issues of interest
tab6 = plot_special_interest(df)
p6 = Panel(child = tab6, title = "Negative Tweets - Special Issues")


# In[ ]:

#Instantiate word distributions
tab7 = draw_word_distribution(df)
p7 = Panel(child = tab7, title = "Unique Word Distribution")


# In[ ]:
#Create a sample of data
data = df.sample(1000)
bokeh_table = data[['user_name','clean', 'label','user_location', 'region', ]]


Columns = [TableColumn(field=Ci, title=Ci) for Ci in bokeh_table.columns] # bokeh columns
data_table = DataTable(columns=Columns, source=ColumnDataSource(bokeh_table), width=1000, height=700) # bokeh table

p8 = Panel(child =data_table, title = 'SampleTable' )


# In[ ]:




# In[ ]:
#Create tabs of all plots
tab = Tabs(tabs =[p1, p2,p3,p4,p5,p6,p7,p8 ])


# In[ ]:
#Save as index
save(tab, filename = 'index.html', title = 'Nigerian Elections 2019')


# In[ ]:
#Check count of unique usernames.
print(len(df.user_name.unique()))


# In[ ]:




# In[ ]:



