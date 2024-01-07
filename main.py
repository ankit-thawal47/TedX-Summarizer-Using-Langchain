import streamlit as st
from utils import load_youtube_video, split_text, initialise_llm, initialize_summarize_chain, get_summary,initialise_plan_chain,generate_plan
from prompts import PROMPT_MAIN,PROMPT_PLANNING,PROMPT_REFINED

#openai api needs to be added here
openai_api_key="xyz"

with st.container():
    st.markdown("""
                PROJECT TEDX SUMMARY AND FUTURE PLAN

                The project will take the youtube video URL of the TedX talk you like and will give you a summary of it.
                At the same time, it will explain the concepts that are hard to digest for a normal person and 
                provide the plan to research it further.

                Made by Ankit.
                """)
    
    input_url = st.text_input(label="URL Input", label_visibility="collapsed",placeholder="Enter URL",key="url_input")
    #input_question = st.text_input(label="Question",key="question")
    submit_button = st.button(label="Submit")

    if submit_button:
        if input_url == 'empty':
            st.write("Please enter correct URL")
            st.stop()
        else:
            with st.spinner("Loading the YouTube transcript....."):
                data = load_youtube_video(input_url)
                docs = split_text(data,chunk_size=1000,chunk_overlap=100)
            
            llm = initialise_llm(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature="0.4")

            with st.spinner("Generating summary...."):
                #question_to_ask = input_question
                #print(question_to_ask)
                summarize_chain_model = initialize_summarize_chain(llm,chain_type="refine",question_prompt=PROMPT_MAIN,refine_prompt=PROMPT_REFINED) 
                summary = get_summary(summarize_chain_model,docs)
            
            print("SUMMARY GENERATED")

            with st.spinner("Explaining and creating a plan..."):
                plan_chain = initialise_plan_chain(llm,prompt=PROMPT_PLANNING,verbose=True)
                plan = generate_plan(plan_chain,summary)
                st.write(plan)
 


