import re

text = """  This string  has a lot    of     spaces for   some  reason. """
clean_text = re.sub(r'\s+', ' ', text).strip()
print(clean_text)



text2 = "This is a &lt; sign."
pattern = r'&([a-zA-Z]+);?'
entity_map = {
    "amp": "&",
    "lt": "<"
}

def replace_entity(match):
    entity = match.group(1)
    return entity_map.get(entity)
cleaned_text2 = re.sub(pattern, replace_entity, text2)
print(cleaned_text2)


text3 = "This is a String."
stop_words = [
    "the", "is", "in", "and", "to", "a", "of", "that", "it", "on","for", "as", "with", "was", "at", "by", "an", "be", "this", "from", "am"]
lafz = text3.lower().split()
filter_words = [word for word in lafz if word not in stop_words]
removed_stop_words = ' '.join(filter_words)
print(removed_stop_words)

# text4 = "thsi is a string"
# tokens = text4.split()
# print (tokens)


text4 = "i love computer and to do computing and computer can computes"
stem_words = [re.sub(r'(ing|er|es)', '', word) for word in text4.split()]
stem_text = ' '.join(stem_words)
print(stem_text)
