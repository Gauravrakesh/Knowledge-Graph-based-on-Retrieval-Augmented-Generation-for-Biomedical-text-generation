'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

import sys
sys.path.append('/d/IIT DELHI/llm_assignment1/LLM2401-Assignment')


from kg_rag.utility import *
import sys
# import retrieve_domain_knowledge_from_wikipedia
# import retrieve_domain_knowledge_from_llm


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False





############



import wikipediaapi
import openai


def retrieve_domain_knowledge_from_wikipedia(question, top_k=2):
    """
    Retrieve relevant domain knowledge from Wikipedia based on keywords in the question.
    
    Parameters:
    question (str): The question for which to retrieve domain knowledge.
    top_k (int): The maximum number of Wikipedia articles to return.
    
    Returns:
    str: A concatenated summary of relevant Wikipedia articles.
    """
    # Define the user agent string as per Wikipedia's policy
    user_agent = "KG-RAG-Biomedical/1.0 (contact: your_project_email@example.com)"
    
    # Initialize the Wikipedia API with the user agent
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
    
    keywords = question.split()[:3]  # Simplified keyword extraction
    
    relevant_snippets = []
    for keyword in keywords:
        page = wiki_wiki.page(keyword)
        if page.exists():
            # Get the summary of the Wikipedia page
            relevant_snippets.append(page.summary)
            if len(relevant_snippets) >= top_k:
                break
    
    # Join the summaries into a single text
    return " ".join(relevant_snippets) if relevant_snippets else "No relevant Wikipedia information found."



def retrieve_domain_knowledge_from_llm(question):
    """
    Retrieve domain knowledge by generating context with an LLM.
    
    Parameters:
    question (str): The question for which to generate domain knowledge.
    
    Returns:
    str: A generated response from the LLM containing domain knowledge.
    """
    prompt = f"Provide relevant background information for understanding the following question:\n\n{question}\n\nPlease include any important context, definitions, or related concepts that would help clarify the question."

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo" based on your usage
        messages=[{"role": "system", "content": "You are a knowledgeable assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=150  # Adjust tokens based on your requirements
    )

    return response['choices'][0]['message']['content']




###########


# MODE = "0"
MODE = "3"
### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    
    for index, row in tqdm(question_df.iterrows(), total=306):
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ### 
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                # MODE 1: JSONLize the context from KG search
                context = retrieve_context(
                    row["text"], vectorstore, embedding_function_for_context_retrieval, 
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, 
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                context_json = json.dumps({"context": context})
                enriched_prompt = f"Context: {context_json}\nQuestion: {question}"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "2":
                # MODE 2: Add prior domain knowledge as suffix to context
                context = retrieve_context(
                    row["text"], vectorstore, embedding_function_for_context_retrieval, 
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, 
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                # Retrieve relevant domain knowledge from Wikipedia
                domain_knowledge = retrieve_domain_knowledge_from_wikipedia(question)
                
                enriched_prompt = f"Context: {context}\nDomain Knowledge: {domain_knowledge}\nQuestion: {question}"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            
            if MODE == "3":
                context = retrieve_context(
                    row["text"], vectorstore, embedding_function_for_context_retrieval, 
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, 
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                # Retrieve relevant domain knowledge from Wikipedia
                domain_knowledge = retrieve_domain_knowledge_from_wikipedia(question)
                
                context_json = json.dumps({"context": context})
                enriched_prompt = f"Domain Knowledge: {domain_knowledge}\nContext: {context_json}\nQuestion: {question}"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            answer_list.append((row["text"], row["correct_node"], output))
        # except Exception as e:
        #     print("Error in processing question: ", row["text"])
        #     print("Error: ", e)
        #     answer_list.append((row["text"], row["correct_node"], "Error"))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error type:", type(e).__name__)
            print("Error details:", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True) 
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    
    main()


