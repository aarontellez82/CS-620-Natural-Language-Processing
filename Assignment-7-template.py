# ***Aaron Tellez***
# Purpose: Assignment-7 - Domain Specific Model Preparation. In this
#          assignment, you will prepare a domain-specific model using data
#          sourced from the minerals dataset. You will select 5 minerals 
#          from the Wikipedia dataset and prepare the data for training 
#          a domain-specific model. You will implement the necessary code
#          to extract the raw data from the dataset, clean and preprocess the
#          data so that it can be used for training a domain specific model.
# 
# Application Requirements:
#      1. Implement the identification and annotation steps for the minerals
#         you have selected. Add annotations for the following entity schema:
#            a. MINERAL_NAME (Hematite, Quartz, etc.)
#            b. MINERAL_PROPERTY (color, hardness, etc.)
#            c. MINERAL_USE (jewelry, construction, etc.)
#            d. TEMPERATURE
#            e. DATE
#            f. MINERAL_DEPOSIT_LOCATION
#            g. MINERAL_STATE (solid, liquid, gas, crystal, etc.)
#            h. MINERAL_COMPOSITION (chemical composition of the mineral)
#      2. Test the data for the minerals against spacy's large model. Verify
#         that the existing model does not identify the entities that you 
#         have annotated.
#      3. Serialize the annotated data into a DocBin format
#      4. Run a space model training cycle
#      5. Test the trained model against a subset of the data that was not used for training.
#      6. Modify annotations in training data accordingly and retrain the model until you are
#         satisfied with the results.
#      7. Retest the model against the test data.
#
# Expected Results:
#      1. You will have generated a domain specific trained model for the minerals 
#         selected from the wikipedia dataset.
#
#
# Version        Author          Date              Description
###############################################################################
#    1           fjm             20-Feb-2026        Initial Template
###############################################################################

##
## Imports for handling the dataset and creating a dataframe
##
## from datasets import load_dataset
## from pandas import DataFrame

## Import Spacy for tokenization.
import spacy
from spacy.tokens import Doc
from spacy.tokens import DocBin
from spacy.language import Language
from spacy.training import Example  # Import Example wrapper for spaCy training updates
from spacy.matcher import PhraseMatcher
from spacy.util import minibatch, compounding
import re
import random

## Import for TF-IDF Vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer

## For topic modeling 
import gensim
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models import LdaModel

## For transformer-based NLP
from transformers import BertTokenizer, BertModel

## Enable or disable debug information
DEBUG1=True

## Register the custom extension attribute on the document
## to support the custom tokenizer component.
if not (Doc.has_extension("filtered_tokens")):
    Doc.set_extension("filtered_tokens", default=None)

## 
## Define custom component for the tokenizer to remove stop words
## and punctuation.
##
@Language.component("snhu_tokenizer")
def snhu_tokenizer(doc):
    ##
    ## Filter out tokens with the following characteristics:
    ##    1. Stop Words
    ##    2. Punctuation
    ##
    filtered_tokens = [ token for token in doc if ((not token.is_stop) and (not token.is_punct) and (not token.is_space)) ]
	
    ## Assign to the registered extension attribute of the doc
    doc._.filtered_tokens = filtered_tokens
	
    ## Return the processed doc
    return doc

# Load trained pipeline from Module 7 directory
try:
    nlp = spacy.load("/home/ubuntu/Desktop/Data/Module-7/minerals_tokenized.pipeline")  # Load domain pipeline from Data directory
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback pipeline

# Register custom component into loaded pipeline
if "snhu_tokenizer" not in nlp.pipe_names:
    nlp.add_pipe("snhu_tokenizer", last=True)  # Add snhu_tokenizer as final component

# Load preprocessed mineral documents from .spacy DocBin file
dataset_docs = []
try:
    doc_bin = DocBin().from_disk("/home/ubuntu/Desktop/Data/Module-7/minerals_tokenized.spacy")  # Read dataset DocBin file
    dataset_docs = list(doc_bin.get_docs(nlp.vocab))  # Load document collection into list
    if DEBUG1:
        print(f"Loaded {len(dataset_docs)} documents from minerals_tokenized.spacy.")  # Print dataset document count
except Exception as e:
    if DEBUG1:
        print(f"Dataset load skipped or unavailable: {e}")  # Handle file loading errors


# Define 5 selected target minerals and expanded gazetteer for entity extraction
TARGET_MINERALS = ["Hematite", "Quartz", "Malachite", "Corundum", "Halite"]

gazetteer = {
    "MINERAL_NAME": ["Hematite", "haematite", "Quartz", "Malachite", "Corundum", "Halite"],
    "MINERAL_STATE": ["crystalline", "crystals", "solid", "crystal structure", "rhombohedral lattice"],
    "MINERAL_COMPOSITION": ["iron oxide", "Fe2O3", "silicon dioxide", "SiO2", "copper carbonate hydroxide", "aluminum oxide", "sodium chloride"],
    "MINERAL_USE": ["jewelry", "decorative jewelry", "construction", "hardstone carvings", "steel production", "rock salt"],
    "MINERAL_PROPERTY": ["hard", "heavy", "green", "red", "brown"],
    "MINERAL_DEPOSIT_LOCATION": ["Brazil", "Australia", "Egypt", "Arkansas", "rocks", "soils"],
    "TEMPERATURE": ["high temperature", "200�C"],
}

# Construct PhraseMatcher for rule-based matching over Wikipedia sentences
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for label, terms in gazetteer.items():
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add(label, patterns)

def annotate_text(text):
    """Automatically extracts entities matching our custom schema with token alignment."""
    doc = nlp.make_doc(text)
    matches = matcher(doc)
    spans = []
    
    # Match dates via regex
    for match in re.finditer(r"\b(1\d{3}|20\d{2})\b", text):
        start, end = match.span()
        span = doc.char_span(start, end, label="DATE")
        if span is not None:
            spans.append(span)
            
    # Add matcher hits
    for match_id, start, end in matches:
        label_name = nlp.vocab.strings[match_id]
        span = doc[start:end]
        spans.append(spacy.tokens.Span(doc, start, end, label=label_name))
        
    filtered_spans = spacy.util.filter_spans(spans)
    annotations = [(span.start_char, span.end_char, span.label_) for span in filtered_spans]
    return annotations

# Build dynamic training data by scanning Wikipedia dataset for our 5 selected minerals
extracted_training_data = []
for doc in dataset_docs:
    doc_full = nlp(doc.text)
    for sent in doc_full.sents:
        sent_text = sent.text.strip()
        if any(mineral.lower() in sent_text.lower() for mineral in TARGET_MINERALS):
            annots = annotate_text(sent_text)
            if annots:
                extracted_training_data.append((sent_text, annots))

# Targeted synthetic training set reinforcing location names and multi-word phrases
synthetic_training_data = [
    ("Quartz is a hard, crystalline mineral composed of silicon dioxide.", [(0, 6, "MINERAL_NAME"), (12, 16, "MINERAL_PROPERTY"), (18, 29, "MINERAL_STATE"), (50, 65, "MINERAL_COMPOSITION")]),
    ("Quartz crystals are widely used in decorative jewelry and hardstone carvings.", [(0, 6, "MINERAL_NAME"), (7, 15, "MINERAL_STATE"), (35, 53, "MINERAL_USE"), (58, 76, "MINERAL_USE")]),
    ("Hematite is a heavy iron oxide mineral found in Brazil, Australia, and Arkansas.", [(0, 8, "MINERAL_NAME"), (14, 19, "MINERAL_PROPERTY"), (20, 30, "MINERAL_COMPOSITION"), (48, 54, "MINERAL_DEPOSIT_LOCATION"), (56, 65, "MINERAL_DEPOSIT_LOCATION"), (71, 79, "MINERAL_DEPOSIT_LOCATION")]),
    ("Hematite was heavily mined in 1945 for construction in Egypt.", [(0, 8, "MINERAL_NAME"), (30, 34, "DATE"), (39, 51, "MINERAL_USE"), (55, 60, "MINERAL_DEPOSIT_LOCATION")]),
    ("Malachite is a green solid copper carbonate hydroxide mineral found in Egypt.", [(0, 9, "MINERAL_NAME"), (15, 20, "MINERAL_PROPERTY"), (21, 26, "MINERAL_STATE"), (27, 53, "MINERAL_COMPOSITION"), (70, 75, "MINERAL_DEPOSIT_LOCATION")]),
    ("Malachite melts at high temperature near 200�C in laboratory tests.", [(0, 9, "MINERAL_NAME"), (19, 35, "TEMPERATURE")]),
    ("Corundum is an extremely solid aluminum oxide mineral.", [(0, 8, "MINERAL_NAME"), (25, 30, "MINERAL_STATE"), (31, 45, "MINERAL_COMPOSITION")]),
    ("Halite is commonly known as rock salt and forms cubic crystals since 1850.", [(0, 6, "MINERAL_NAME"), (28, 37, "MINERAL_USE"), (48, 62, "MINERAL_STATE"), (69, 73, "DATE")])
]

training_data = extracted_training_data + synthetic_training_data

# Partitioned test set for testing custom entity extraction
testing_data = [
    (
        "Quartz crystals are formed at high temperature in pegmatites located in Arkansas.",
        [(0, 6, "MINERAL_NAME"), (7, 15, "MINERAL_STATE"), (30, 46, "TEMPERATURE"), (72, 80, "MINERAL_DEPOSIT_LOCATION")]
    ),
    (
        "Hematite is an iron oxide used in construction since 1850.",
        [(0, 8, "MINERAL_NAME"), (15, 25, "MINERAL_COMPOSITION"), (34, 46, "MINERAL_USE"), (53, 57, "DATE")]
    ),
    (
        "Malachite is a solid mineral prized for decorative jewelry in Egypt.",
        [(0, 9, "MINERAL_NAME"), (15, 20, "MINERAL_STATE"), (40, 58, "MINERAL_USE"), (62, 67, "MINERAL_DEPOSIT_LOCATION")]
    )
]

# Step 2 Verification: Check existing model against target entities
if DEBUG1:
    print("--- Baseline Model Verification (spacy pretrained) ---")
    for text, _ in testing_data:
        doc = nlp(text)  # Run existing pipeline on test input
        print(f"Text: {text}")
        print(f"Entities Found: {[(ent.text, ent.label_) for ent in doc.ents]}\n")  # Print entity outputs

# Function to write annotated sentence lists to serialized DocBin files
def save_docbin(dataset, filepath):
    db = DocBin()  # Create DocBin container instance
    for text, annotations in dataset:
        doc = nlp.make_doc(text)  # Create document object from text
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)  # Convert character offsets to span
            if span is not None:
                ents.append(span)  # Collect valid span objects
        doc.ents = ents  # Assign entity spans to document
        db.add(doc)  # Insert doc into container
    db.to_disk(filepath)  # Save container to file path

save_docbin(training_data, "./training_data.spacy")  # Write training set to disk
save_docbin(testing_data, "./testing_data.spacy")  # Write testing set to disk

# Construct and run custom NER training loop
ner_nlp = spacy.blank("en")  # Initialize blank English language pipeline
ner = ner_nlp.add_pipe("ner")  # Create new named entity recognizer

# Register all labels
for _, annotations in training_data:
    for _, _, label in annotations:
        ner.add_label(label)

train_examples = []
for text, annotations in training_data:
    doc = ner_nlp.make_doc(text)  # Generate doc object
    valid_annots = []
    for s, e, l in annotations:
        span = doc.char_span(s, e, label=l)
        if span is not None:
            valid_annots.append((span.start_char, span.end_char, l))
    train_examples.append(Example.from_dict(doc, {"entities": valid_annots}))

optimizer = ner_nlp.initialize()  # Initialize neural weights

# Train over batch compounding loop with 40 iterations for complete convergence
for iteration in range(40):
    random.shuffle(train_examples)
    losses = {}
    batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        ner_nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)

ner_nlp.to_disk("./custom_minerals_model")  # Save trained domain model directory

## Display results against testing dataset
print("--- Trained Custom Domain Model Results ---")
custom_model = spacy.load("./custom_minerals_model")  # Load custom model pipeline
for text, expected in testing_data:
    doc = custom_model(text)  # Execute custom model inference
    print(f"Test Input: '{text}'")
    print(f"Detected Custom Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")  # Display extracted custom labels
    print(f"Expected Ground Truth:   {[(text[s:e], lbl) for s, e, lbl in expected]}\n")  # Display original target labels