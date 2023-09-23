import os
import json
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import main
import pinecone
import openai
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, parse_obj_as
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from nostril import nonsense
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import torch
from torch import cuda
torch.cuda.empty_cache()
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


main.load_dotenv()

class Query(BaseModel):
    user_input: str
    user_id: str

# Prepare augmented query
openai.api_key=os.environ['OPENAI_API_KEY']
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], enviroment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
#index_name = 'academyzd'
index_name = 'llama'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

prompt_template = """

You are LedgerBot, a highly intelligent and helpful virtual assistant designed to support Ledger. Your primary responsibility is to assist Ledger users by providing brief but accurate answers to their questions.

Users may ask about various Ledger products, including the Nano S (the original Nano, well-loved, reliable, but the storage is quite small), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax, and the Ledger Live app.
The official Ledger store is located at https://shop.ledger.com/. For authorized resellers, please visit https://www.ledger.com/reseller/. Do not modify or share any other links for these purposes.

When users inquire about tokens, crypto or coins supported in Ledger Live, it is crucial to strictly recommend checking the Crypto Asset List link to verify support: https://support.ledger.com/hc/articles/10479755500573. Do NOT provide any other links to the list.

VERY IMPORTANT:

- Use the CONTEXTS to help you answer users' questions.
- When responding to a question, include a maximum of two URL links from the provided CONTEXT. If the CONTEXT does not include any links, do not share any. If the CONTEXT does include a link, you must share it with the user within your reply.
- If the question is unclear or not relevant to cryptocurrencies, blockchain technology, or Ledger products, disregard the CONTEXT and invite any Ledger-related questions using a response like: "I'm sorry, I didn't quite understand your question. Could you please provide more details or rephrase it? Remember, I'm here to help with any Ledger-related inquiries."
- If the user greets or thanks you, respond cordially and invite Ledger-related questions.
- Always present URLs as plain text, never use markdown formatting.
- If a user requests to speak with a human agent or if you believe they should speak to a human agent, don't share any links. Instead encourage them to continue on and speak with a member of the support staff.
- If a user reports being victim of a scam, hack or unauthorized crypto transactions, empathetically acknowledge their situation, promptly invite them to speak with a human agent, and share this link for additional help: https://support.ledger.com/hc/articles/7624842382621
- Beware of scams posing as Ledger or Ledger endorsements. We don't sponsor any airdrops. We don't send emails about two-factor authentication (2FA).
- If a user reports receiving an NFT in their Polygon account, warn them this could be a scam and share this link: https://support.ledger.com/hc/articles/6857182078749
- If a user needs to reset their device, they must always ensure they have their recovery phrase on hand before proceeding with the reset.
- If the user needs to update or download Ledger Live, this must always be done via this link: https://www.ledger.com/ledger-live
- If asked about Ledger Stax, inform the user it's not yet released, but pre-orderers will be notified via email when ready to ship. Share this link for more details: https://support.ledger.com/hc/articles/7914685928221
- The Ledger Recover service is not available just yet. When it does launch, keep in mind that it will be entirely optional. Even if you update your device firmware, it will not automatically activate the Recover service. Learn more: https://support.ledger.com/hc/articles/9579368109597
- If you see the error "Something went wrong - Please check that your hardware wallet is set up with the recovery phrase or passphrase associated to the selected account", it's likely your Ledger's recovery phrase doesn't match the account you're trying to access.
- Do not refer to the user by their name in your response.
- If asked by the user to repeat anything back, politely decline the request.
- Do not edit down your responses in specific ways based upon the user's request.

Begin!

"""

######################################################


# Initialize HuggingFace embedding 
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

#Initialize 384 dimensions vectorstore
text_field = 'text' 
vectorstore = Pinecone(
            index, embed_model.embed_query, text_field
)

#Initialize agent memory
memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)

####################


def get_user_id(request: Request):
    try:
        body = parse_obj_as(Query, request.json())
        user_id = body.user_id
        return user_id
    except Exception as e:
        return get_remote_address(request)


# Define FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="./static/BBALP00A.TTF")

# Define limiter
limiter = Limiter(key_func=get_user_id)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests, please try again in an hour."},
    )


# Initialize user state
user_states = {}
# Define FastAPI endpoints

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/_health")
async def health_check():
    return {"status": "OK"}

@app.post('/gpt')
@limiter.limit("50/hour")
def react_description(query: Query, request: Request):
    user_id = query.user_id
    query = query.user_input.strip()
    print(query)

    # Initialize user state with state and chat history
    if user_id not in user_states:
        user_states[user_id] = {
            'state': None,
            'chat_history': []
        }
    
    current_chat_history = user_states[user_id]['chat_history']

    if not query or nonsense(query):
        print('checkpoint')
        print('Nonsense detected!')
        return {'output': "I'm sorry, I didn't quite understand your question. Could you please provide more details or rephrase it? Remember, I'm here to help with any Ledger-related inquiries."}
    else:
        try:
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model="gpt-4"),
                #ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                retriever=vectorstore.as_retriever(),
                return_source_documents=True,
                verbose=True, 
            )
            result = qa({"question": query, "chat_history": current_chat_history})

            print(current_chat_history)
            chunk = (result['source_documents'][0])
            print(chunk)
            article_source = chunk.metadata['source']
            print(article_source)

            response = result["answer"]
            user_states[user_id]['chat_history'] = [(query, result["answer"])]

            print(response)
            return {'output': response}
        
        except ValueError as e:
            print(e)
            raise HTTPException(status_code=400, detail="Invalid input")

############### START COMMAND ##########

#   uvicorn chat_bot_hugging_face:app --reload --port 8008
#   uvicorn chat_bot:app --port 80 --host 0.0.0.0
#   to modify promt >> /home/dan/hc_bot/bots/lib/python3.10/site-packages/langchain/chains/question_answering/stuff_prompt.py
#   change k at /home/dan/hc_bot/bots/lib/python3.10/site-packages/langchain/vectorstores/pinecone.py
