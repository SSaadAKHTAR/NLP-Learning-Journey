import re
# Task 1.1
text = "Call me at 0321-4567890 or (021) 9876543. Office: +92 333 1112233"
pattern = r'(\+?\d{1,3}[-\s]?)?(\(?\d{2,4}\)?[-\s]?)?\d{3,4}[-\s]?\d{4}'
redacted_text = re.sub(pattern, "[REDACTED]", text)

print("Original", text)
print("Redacted", redacted_text)


# Task 1.2
text2 = "My birthday is 05/21/2002 and project deadline is 12/07/2025."
pattern = r'(\b\d{2})/(\d{2})/(\d{4}\b)'
reformatted_text = re.sub(pattern, r'\3-\1-\2', text2)

print("Original:", text2)
print("Reformatted:", reformatted_text)


# Task 1.3
text3 = """This   is   a     messy\tstring\nwith   lots   of   
spaces"""
clean_text = re.sub(r'\s+', ' ', text3).strip()

print("Original:", repr(text3))
print("Cleaned:", repr(clean_text))


# Task 2
text4 = """
I have 42 apples and 3 oranges.
The price is $42 or sometimes 42.99 per kg.
Also, I sold 100 items today.
"""
pattern = r'(?<!\$)(?<!\d\.)\b\d+\b(?!\.\d)'
matches = re.findall(pattern, text4)
print("Matches found:", matches)