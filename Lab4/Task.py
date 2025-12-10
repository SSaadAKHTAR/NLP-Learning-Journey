import spacy 
from spacy import displacy 

nlp = spacy.load("en_core_web_sm")

text = 'Nvidia is a gpu design company.'
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
displacy.render(doc, style='ent')



