import nltk
import spacy
from nltk.stem import WordNetLemmatizer, PorterStemmer

###########################
##### NLTK FUNCTIONS ######
###########################

# Function for NLTK tokenization
def nltk_tokenize(text):
    return nltk.word_tokenize(text)

# Function for NLTK lemmatization a text 
def nltk_lemmatize(text):
    # Initialize the NLTK lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize each token
    return lemmas

# Function for NLTK lemmatization on a set of tokens
def nltk_lemmatize_tokens(token_set):
    lemmatizer = WordNetLemmatizer()
    return {lemmatizer.lemmatize(token) for token in token_set}

# For text
def nltk_stemming(text):
    porter_stemmer = PorterStemmer()
    words = text.split()
    return [porter_stemmer.stem(word) for word in words]

# For set
# Function for NLTK stemming of a set
def nltk_set_stemming(tokens):
    porter_stemmer = PorterStemmer()
    return {porter_stemmer.stem(word) for word in tokens}  # Apply stemming and return as set

# Function to tokenize all messages in spam_df using NLTK
def create_token_set_nltk(dataframe):
    """
    Tokenizes all messages in the dataframe using NLTK and returns a unique set of tokens.
    """
    all_tokens = set()
    for message in dataframe['message']:
        tokens = nltk_tokenize(message)  # Use the NLTK tokenization function
        all_tokens.update(tokens)
    return all_tokens


###########################
##### spaCy FUNCTIONS ######
###########################


# Function for spaCy tokenization
def spacy_tokenize(text, nlp):
   # nlp = spacy.load("en_core_web_sm")  # Load spaCy model
    doc = nlp(text)
    return [token.text for token in doc]



# Function for spaCy lemmatization a test
def spacy_lemmatize(text, nlp):
    #nlp = spacy.load("en_core_web_sm")  # Load spaCy model
    doc = nlp(text)  # Process the text using spaCy
    lemmas = [token.lemma_ for token in doc]  # Extract lemmatized tokens
    return lemmas


# Function for spaCy lemmatization on a set of tokens
def spacy_lemmatize_tokens(token_set, nlp):
    return {token.lemma_ for token in nlp(" ".join(token_set))}




# Function to tokenize all messages in spam_df using spaCy
def create_token_set_spacy(dataframe, nlp):
    """
    Tokenizes all messages in the dataframe using spaCy and returns a unique set of tokens.
    """
    all_tokens = set()
    for message in dataframe['message']:
        tokens = spacy_tokenize(message, nlp)  # Use the spaCy tokenization function
        all_tokens.update(tokens)
    return all_tokens


##############################################

# Perform lemmatization on both nltk and spacy
def create_lemma_set(nltk_tokens_set, spacy_tokens_set, nlp):
   
    # Lemmatize with NLTK
    nltk_lemmas_set = nltk_lemmatize_tokens(nltk_tokens_set)

    # Lemmatize with spaCy
    spacy_lemmas_set = spacy_lemmatize_tokens(spacy_tokens_set, nlp)

    return nltk_lemmas_set, spacy_lemmas_set


# def find_message_to_remove_stemm_less_lemma_equal(spam_df):
#     # Store the initial counts of stemmed and lemmatized tokens for comparison
#     all_nltk_tokens = set()
#     all_nltk_lemmas = set()
    
#     # Tokenize and lemmatize the entire dataset
#     for message in spam_df['message']:
#         nltk_tokens = nltk_tokenize(message)
        
#         # Apply lemmatization
#         nltk_lemmas = nltk_lemmatize(message)
        
#         # Apply stemming
#         nltk_stems = nltk_set_stemming(nltk_tokens)

#         # Update the sets with tokens, lemmas, and stems
#         all_nltk_tokens.update(nltk_tokens)
#         all_nltk_lemmas.update(nltk_lemmas)
    
#     # Now loop over the messages and check the difference when a message is removed
#     for index, message in spam_df.iterrows():
#         #print(f"Checking message {index}...")

#         # Create sets for tokens, lemmas, and stems of the current message
#         nltk_tokens = nltk_tokenize(message['message'])
#         nltk_lemmas = nltk_lemmatize(message['message'])
#         nltk_stems = nltk_set_stemming(nltk_tokens)

#         # Calculate the token sets after removing the current message
#         temp_nltk_tokens = all_nltk_tokens - set(nltk_tokens)
#         temp_nltk_lemmas = all_nltk_lemmas - set(nltk_lemmas)
#         temp_nltk_stems = all_nltk_tokens - set(nltk_stems)

#         # Check if removing the current message results in less stemmed tokens but no change in lemmatized tokens
#         if len(temp_nltk_stems) < len(all_nltk_tokens) and len(temp_nltk_lemmas) == len(all_nltk_lemmas):
#             print(f"Message at index {index} causes a change in stemmed tokens but not lemmatized tokens.")
#             return message['message']  # Return the message that causes the change

#     print("No message found that satisfies the condition.")
#     return None

def find_message_to_remove_stemm_less_lemma_equal(spam_df):
    # Store the initial counts of stemmed and lemmatized tokens for comparison
    all_nltk_tokens = set()
    all_nltk_lemmas = set()
    all_nltk_stems = set()  # We need a separate set for stemmed tokens
    
    # Tokenize, lemmatize, and stem the entire dataset
    for message in spam_df['message']:
        nltk_tokens = nltk_tokenize(message)
        
        # Apply lemmatization
        nltk_lemmas = nltk_lemmatize(message)
        
        # Apply stemming
        nltk_stems = nltk_set_stemming(nltk_tokens)

        # Update the sets with tokens, lemmas, and stems
        all_nltk_tokens.update(nltk_tokens)
        all_nltk_lemmas.update(nltk_lemmas)
        all_nltk_stems.update(nltk_stems)  # Update stemmed tokens set
    
    # Now loop over the messages and check the difference when a message is removed
    for index, message in spam_df.iterrows():
        # Create sets for tokens, lemmas, and stems of the current message
        nltk_tokens = nltk_tokenize(message['message'])
        nltk_lemmas = nltk_lemmatize(message['message'])
        nltk_stems = nltk_set_stemming(nltk_tokens)

        # Calculate the token sets after removing the current message
        #temp_nltk_tokens = all_nltk_tokens - set(nltk_tokens)
        temp_nltk_lemmas = all_nltk_lemmas - set(nltk_lemmas)
        temp_nltk_stems = all_nltk_stems - set(nltk_stems)  # Corrected to subtract from stemmed tokens


        # Check if removing the current message results in less stemmed tokens but no change in lemmatized tokens
        if len(temp_nltk_stems) < len(all_nltk_stems) and len(temp_nltk_lemmas) == len(all_nltk_lemmas):
            print(f"Message at index {index} causes a change in stemmed tokens but not lemmatized tokens.")
            return message['message']  # Return the message that causes the change

    print("No message found that satisfies the condition.")
    return None


def find_message_to_remove_lemma_less_stemm_equal(spam_df):
    # Store the initial counts of lemmatized and stemmed tokens for comparison
    all_nltk_lemmas = set()
    all_nltk_stems = set()  # We need a separate set for stemmed tokens
    
    # Lemmatize and stem the entire dataset
    for message in spam_df['message']:
        nltk_tokens = nltk_tokenize(message)
        
        # Apply lemmatization
        nltk_lemmas = nltk_lemmatize(message)
        
        # Apply stemming
        nltk_stems = nltk_set_stemming(nltk_tokens)

        # Update the sets with lemmas and stems
        all_nltk_lemmas.update(nltk_lemmas)
        all_nltk_stems.update(nltk_stems)
    
    # Now loop over the messages and check the difference when a message is removed
    for index, message in spam_df.iterrows():
        # Create sets for lemmas and stems of the current message
        nltk_tokens = nltk_tokenize(message['message'])
        nltk_lemmas = nltk_lemmatize(message['message'])
        nltk_stems = nltk_set_stemming(nltk_tokens)

        # Calculate the sets after removing the current message
        temp_nltk_lemmas = all_nltk_lemmas - set(nltk_lemmas)
        temp_nltk_stems = all_nltk_stems - set(nltk_stems)

        # Check if removing the current message results in fewer lemmatized tokens but no change in stemmed tokens
        if len(temp_nltk_lemmas) < len(all_nltk_lemmas) and len(temp_nltk_stems) == len(all_nltk_stems):
            print(f"Message at index {index} causes a change in lemmatized tokens but not in stemmed tokens.")
            return message['message']  # Return the message that causes the change

    print("No message found that satisfies the condition.")
    return None
