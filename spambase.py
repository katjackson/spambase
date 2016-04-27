import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
# I didn't use these today, but I want to remember them
# from sklearn.linear_model import LinearRegression
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures

# Reading in data with column names
columnsx = ['word_freq_make',
            'word_freq_address',
            'word_freq_all',
            'word_freq_3d',
            'word_freq_our',
            'word_freq_over',
            'word_freq_remove',
            'word_freq_internet',
            'word_freq_order',
            'word_freq_mail',
            'word_freq_receive',
            'word_freq_will',
            'word_freq_people',
            'word_freq_report',
            'word_freq_addresses',
            'word_freq_free',
            'word_freq_business',
            'word_freq_email',
            'word_freq_you',
            'word_freq_credit',
            'word_freq_your',
            'word_freq_font',
            'word_freq_000',
            'word_freq_money',
            'word_freq_hp',
            'word_freq_hpl',
            'word_freq_george',
            'word_freq_650',
            'word_freq_lab',
            'word_freq_labs',
            'word_freq_telnet',
            'word_freq_857',
            'word_freq_data',
            'word_freq_415',
            'word_freq_85',
            'word_freq_technology',
            'word_freq_1999',
            'word_freq_parts',
            'word_freq_pm',
            'word_freq_direct',
            'word_freq_cs',
            'word_freq_meeting',
            'word_freq_original',
            'word_freq_project',
            'word_freq_re',
            'word_freq_edu',
            'word_freq_table',
            'word_freq_conference',
            'char_freq_;',
            'char_freq_(',
            'char_freq_[',
            'char_freq_!',
            'char_freq_$',
            'char_freq_#',
            'capital_run_length_average',
            'capital_run_length_longest',
            'capital_run_length_total',
            'SPAM'
            ]
spambase = pd.read_csv('spambase.data', header=None, names=columnsx)

# Splitting data into test and train samples with sklearn.train_test_split
train, test = train_test_split(spambase, test_size=0.4, train_size=0.6,
                               random_state=60)
y_train = train.SPAM
y_test = test.SPAM
X_train = train.drop('SPAM', 1)
X_test = test.drop('SPAM', 1)

# instantiate and fit a Naive Baye, get scores for training and test data
spambot = MultinomialNB()
spambot.fit(X_train, y_train)
spambot.score(X_train, y_train)
spambot.score(X_test, y_test)

# Many other methods demo'd in notebook, but this was the best score
use_cols = ['word_freq_remove', 'word_freq_3d', 'word_freq_internet',
            'word_freq_free', 'char_freq_!',
            'word_freq_edu', 'word_freq_re', 'word_freq_george',
            'word_freq_hp', 'word_freq_business', 'word_freq_your',
            'word_freq_credit', 'SPAM']
train, test = train_test_split(spambase[use_cols], test_size=0.4,
                               train_size=0.6, random_state=900)
y_train = train.SPAM
y_test = test.SPAM
X_train = train.drop('SPAM', 1)
X_test = test.drop('SPAM', 1)

spammy = MultinomialNB()

spammy.fit(X_train, y_train)
spammy.score(X_test, y_test)
