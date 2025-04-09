
import os
import pandas as pd 
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from flask import Flask, g, render_template, request,jsonify





# Gfinancial_df = pd.DataFrame()
# GtablesList = [] 
# Gdocument_embeddings= []
# Gchunks_text=  []

# Initialize the tokenizer and model
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForQuestionAnswering.from_pretrained(model_name)

# model_Questions = pipeline("question-answering", model=model, tokenizer=tokenizer)

# #initialize embedings 
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# #initiazize translator 
# # Load models and tokenizers for both translation en to es and es to en directions
# #https://huggingface.co/Helsinki-NLP/opus-mt-es-en
# model_name_es_to_en = 'Helsinki-NLP/opus-mt-es-en'
# model_name_en_to_es = 'Helsinki-NLP/opus-mt-en-es'

# tokenizer_es_to_en = MarianTokenizer.from_pretrained(model_name_es_to_en)
# model_es_to_en = MarianMTModel.from_pretrained(model_name_es_to_en)

# tokenizer_en_to_es = MarianTokenizer.from_pretrained(model_name_en_to_es)
# model_en_to_es = MarianMTModel.from_pretrained(model_name_en_to_es)


financial_df = None
file_loaded = False
rag_chain = None

def load_global_data(pathFile):
    if 'financial_df' not in g : #or 'tablesList' not in g or 'document_embeddings' not in g or 'chunks_text' not in g:
        print('load data financial')
        # g.financial_df, g.tablesList, g.document_embeddings, g.chunks_text = getData(pathFile)
        g.financial_df = getData(pathFile)

    return g.financial_df #, g.tablesList, g.document_embeddings, g.chunks_text



def getData(filePath):
    global rag_chain, llm, vectorstore, financial_df, file_loaded
    if filePath == '' :
       


        if not file_loaded:
            financial_df = pd.read_csv('static/data/finnacial_release_df.csv')
            file_loaded = True

        # Convert to LangChain documents
        documents = [
            Document(page_content=row['extracted_text'], metadata={"start_page": row['start_page'], "tag": row['tag']})
            for _, row in financial_df.iterrows()
        ]

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)


        # Split documents into smaller chunks
        doc_chunks = []
        doc_metadatas = []
        for doc in documents:
            # Split the page content into chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            # Append each chunk with its metadata
            for chunk in chunks:
                doc_chunks.append(chunk)
                doc_metadatas.append(doc.metadata)

        #create embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MPnet-base-v2")
        #create entries from FAISS 
        vectorstore = FAISS.from_texts(doc_chunks, embedding_model)

        #initialize the model with mistral 
        #keep a temperature of 0.7 to avoid hallucinations and keep at bay the creativity of the text 
        llm = LlamaCpp(
            model_path="static/data/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Update to your model path
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,
            n_batch=8,
            verbose=True,
        )
        #Add Retreival to Mistral, allowing producing better responses base on the documents provided

        # rag_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        #     return_source_documents=True
        # )

        # Ensure the chain is only created once
        if rag_chain is None:
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )


        # table0 = pd.read_csv('static/data/table_0.csv')
        # table1 = pd.read_csv('static/data/table_1.csv')
        # table2 = pd.read_csv('static/data/table_2.csv')
        # table3 = pd.read_csv('static/data/table_3.csv')
        # table4 = pd.read_csv('static/data/table_4.csv')
        # table5 = pd.read_csv('static/data/table_5.csv')
        # table6 = pd.read_csv('static/data/table_6.csv')
        # table7 = pd.read_csv('static/data/table_7.csv')
        # table8 = pd.read_csv('static/data/table_8.csv')
        # table9 = pd.read_csv('static/data/table_9.csv')

        # tablesList = [table0, table1, table2, table3, table4, table5, table6, table7, table8, table9]


        # Apply the sentence tokenizer to each row in the DataFrame
       # financial_df['sentences'] = financial_df['extracted_text'].apply(utils_finance.split_text_to_sentences)

        # Create one list of all sentences
        #all_sentences = [sentence for sublist in financial_df['sentences'] for sentence in sublist]

        #chuncks of text, extracted from pdf
       #chunks_text = [sublist for sublist in financial_df['extracted_text'] ]

        # Create embeddings for documents
      #  document_embeddings = embedder.encode(chunks_text, convert_to_tensor=False)


    return financial_df, rag_chain #, tablesList, document_embeddings, chunks_text



# #take text no matter the language, detect language and only if is spanish make the translation
# def translate_text(text, model, tokenizer):
#     #make translation to spanish
     
#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
#     # Perform the translation
#     with torch.no_grad():
#         translated = model.generate(**inputs)

#     # Decode the translated text
#     translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
#     return translated_text


# def obtainTranslation(text, target='en'):
#     if  text == '':
#         return '', target
#     # Detect language of the input text
#     language = detect(text)
#     print(text)
#     isInput = target == 'es'
    
#     # Determine if translation is needed
#     shouldTranslate = (language != target)
    
#     # If the language matches the target or translation is not needed
#     if not shouldTranslate and language == target:
#         return text, language

#     # If translation is needed and input language is Spanish
#     if shouldTranslate and language == 'es':
#         result = translate_text(text, model_es_to_en, tokenizer_es_to_en)
#     elif shouldTranslate and language == 'en':
#         result = translate_text(text, model_en_to_es, tokenizer_en_to_es)
#     else:
#         result = text

#     return result, language

    


# # cosine similarity of the embedings and return the best options 
# def retrieve_context(question, document_embeddings, documents, top_k=1):
#     question_embedding = embedder.encode(question, convert_to_tensor=False)
    
#     document_embeddings = torch.tensor(document_embeddings)
    
#     # using pytorch calculate cosine similarity
#     scores = util.pytorch_cos_sim(question_embedding, document_embeddings)[0]
    
#     # Move scores back to CPU and convert to numpy array
#     scores = scores.cpu().numpy()
    
#     # Retrieve the top_k results
#     top_results = np.argpartition(-scores, range(top_k))[0:top_k]
    
#     return [documents[idx] for idx in top_results]

#get response from model, after embeding text and extractign contexxt 
def get_response(question):
    global financial_df, rag_chain  # Make sure we're working with the global variables

    print('before quesion',financial_df, rag_chain )

    # Only load data if rag_chain is not initialized
    if rag_chain is None:
        print('train model' )
        financial_df, rag_chain = load_global_data('')

    print('model fully trainned', rag_chain )

    # Use rag_chain to get the response
    result = rag_chain.invoke({"query": question})

    print('result quesion', result['result'] )

    return result['result']

    # # financial_df, tablesList, document_embeddings, chunks_text = load_global_data('')
    # financial_df, rag_chain = load_global_data('')


    # result = rag_chain({"query": question})
    # return result['result']
    #  #translate question if necesary 
    # result, language = obtainTranslation(question)
    # #take question from translation 
    # question =result
    # print('Question after translation', question,'Language=', language)


    # # Retrieve relevant context base on the consine similarity
    # context = retrieve_context(question, document_embeddings, chunks_text)
    

    # # Answer the question using BertForQuestionAnswering
    # result = model_Questions(question=question, context=" ".join(context))

    # #check data on tables 
    # relevant_Tables= utils_finance.check_allTables_relevan_data(question, tablesList)

    # translatedResult =result['answer']
    # print('\nEnglish Answer', translatedResult,'Language=', language)


    # if language == 'es':

    #     #translate output to language needed 
    #     translatedResult, language = obtainTranslation(result['answer'], language)
    #     print('\n Answer after translation ', translatedResult, language)

    # return translatedResult , relevant_Tables


app = Flask(__name__)
app.static_folder = 'static'

#config folder to read data
app.config['UPLOAD_FOLDER'] = 'static/data'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    userText = request.args.get('msg')

    # result, relevant_Tables = get_response(userText)
    result = get_response(userText)

    print(result)

   
    return  jsonify({
        'result': result,
        'relevant_Tables': ""
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    message = request.form.get('message')
    file = request.files.get('file')

    #load new file 
    # load_global_data()

    
    if file:
        # Process the file here (e.g., save it, read it, etc.)
        filename = secure_filename(file.filename)
        file.save(os.path.join('pdfFiles', filename))
        return jsonify({'response': f'File {filename} uploaded successfully!'})
    
    # Process the text message
    #response = process_user_message(message)
    return "file uploaded!" #jsonify({'response': response})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4700)
    app.run()