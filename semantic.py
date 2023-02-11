# this program compares similarity of words and sentences with 'en_core_web_md' and 'en_core_web_sm' 


'''
- The interesting thing about the similarities between cat, monkey and banana is, 
  not only SpaCy detects cat and monkey are both animal but also it detects that that monkey eats banana

- It seems en_core_web_md is more reliable, and also by using en_core_web_sm and calling similarity we got this Warning message
        "The model you're using has no word vectors loaded, so the result of the Token.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements"

  Based on my research if we need to process large amounts of text data 
  or perform more advanced NLP tasks, we may want to consider using 'en_core_web_md'. 
  If we have limited computational resources, or if we only need to perform basic NLP tasks, 
  then 'en_core_web_sm' might be a better choice. 
'''

import spacy

nlp_md = spacy.load('en_core_web_md')
nlp_sm = spacy.load('en_core_web_sm')

print("-----------------------------------------------")
print("Similarity with en_core_web_md:")

word1 = nlp_md("cat")
word2 = nlp_md("monkey")
word3 = nlp_md("banana")

print(f"{word1}, {word2} => {word1.similarity(word2)}")
print(f"{word3}, {word2} => {word3.similarity(word2)}")
print(f"{word3}, {word1} => {word3.similarity(word1)}")

print()
print("Similarity with en_core_web_sm:")

word1 = nlp_sm("cat")
word2 = nlp_sm("monkey")
word3 = nlp_sm("banana")

print(f"{word1}, {word2} => {word1.similarity(word2)}")
print(f"{word3}, {word2} => {word3.similarity(word2)}")
print(f"{word3}, {word1} => {word3.similarity(word1)}")

print()
print("-----------------------------------------------")
print("Compare a series of words with one another with en_core_web_md:")

tokens = nlp_md('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print()
print("Compare a series of words with one another with en_core_web_sm:")

tokens = nlp_sm('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print()
print("-----------------------------------------------")
print("Similarity between sentences with en_core_web_md:")

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp_md(sentence_to_compare)

for sentence in sentences:
    similarity = nlp_md(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
    
print()
print("Similarity between sentences with en_core_web_sm:")

model_sentence = nlp_sm(sentence_to_compare)

for sentence in sentences:
    similarity = nlp_sm(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

print()
print("-----------------------------------------------")


