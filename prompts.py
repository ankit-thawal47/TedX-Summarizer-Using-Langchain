
from langchain.prompts import PromptTemplate


prompt_template_main = """
You are summarization tool, which takes the data and summarize it in 500 words or less.
The summary text should not loose the essence of the main data, and it is easily understandable for a 15 year old STEM student.
The summary will additionally should address the main question as follow : 
------------
{text}
------------
Create a detailed summary by keeping in mind the above question.
"""

PROMPT_MAIN = PromptTemplate(template=prompt_template_main,input_variables=["text"])

refined_template = """
You are expert in summarizing the text, and can summarize the text in 500 words or less without loosing the core meaning of the text.
The summarized text should be undertandable for a 15 year old STEM student.
We have existing summary of the same text : {existing_answer}.
We have an opportunity to refine the text, so we dont loose the main meaning of the exisisting data, and at the same time, mould the summary
to answer the question:
------------
{text}
------------
Given the context, refine the summary(only if needed). The summary should be as detailed as possible. If it can't be refined further, return the original summary.
"""

PROMPT_REFINED = PromptTemplate(template=refined_template,input_variables=["existing_answer","text"])


further_plan_template = """
You are expert in summarizing the text, and can summarize the text in 500 words or less without loosing the core meaning of the text.
The summarized text should be undertandable for a 15 year old STEM student.
You have to give a detailed plan and timeline for that student to learn the details and different concepts written in the summary : {refined_summary2}.
You have to use the bullet points and be as detailed as possible whilst giving the details.
"""

PROMPT_PLANNING = PromptTemplate(template=further_plan_template,input_variables=["refined_summary2"])