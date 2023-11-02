from transformers import pipeline

model_name = "distilbert-base-uncased-distilled-squad"
output_directory = "/Users/balusasidharanpillai/pro/aimodels"
question_answerer = pipeline("question-answering", modeel=model_name, output_dir=output_directory)

# context = r"""
#               "Hello My name is Balu, I have two kids  and a dog.  " \
#               "My Kids name is Siddharth and Jagannathan. My Mothers name is Usha, My wife name is Soumya." \
#               "live in connecticut. I work for an insurance company. She works for a bio medical company". \
#               "I'm 37 years old. My Wife's age is 35, Kids age is 6 and 8 respectively. " \
#               "I work for Cigna. This is the link to the page http://www.balu.com "\
              
#"""
with open('input.txt', 'r') as file:
    file_contents = file.read()
context = file_contents

while True:
    question = input("Enter Question ")
    result = question_answerer(question=question, context=context, device="mps")
    print(
        f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
