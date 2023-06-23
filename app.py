from os import path, listdir
import streamlit as st
from streamlit_embedcode import github_gist
import streamlit.components.v1 as com
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.collocations import *
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
from os import path, listdir
import glob
import pickle
from pathlib import Path
from plotly import graph_objs as go
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel
from st_material_table import st_material_table
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import streamlit.components.v1 as components
import streamlit.components.v1 as stc



# Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set wide layout
st.set_page_config(
    page_title = 'SMG TOWARDS ZERO TB',
    page_icon = '✅',
    layout = 'wide'
)

 
st.markdown("""
<h1 style='text-align:center;padding: 0px 0px;color:Black;font-size:400%;'>Semarang Towards Zero TB</h1>
<h2 style='text-align:center;padding: 0px 0px;color:Black;font-size:150%;'><b></b></h2>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Tekan enter untuk username dan password")
        return False
    else:
        # Password correct.
        return True

if check_password():
    com.html("""
           <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <html>
            <body>
            <div class='tableauPlaceholder' id='viz1687481020742' style='position: relative'><noscript><a href='#'><img alt='Dashboard Home ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tb&#47;tb234&#47;DashboardHome&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='tb234&#47;DashboardHome' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tb&#47;tb234&#47;DashboardHome&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1687481020742');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1809px';vizElement.style.height='2757px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1809px';vizElement.style.height='2757px';} else { vizElement.style.width='100%';vizElement.style.height='2227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
             </body>
            </html>
            """
            , height=2700)
