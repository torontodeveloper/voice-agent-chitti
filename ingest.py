import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

load_dotenv()


class RAGDataBase:
    def __init__(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.assistants_list = pc.assistant.list_assistants()
        print(f"list of assistants, {self.assistants_list}")
        list_of_assistants = [assistant.name for assistant in self.assistants_list]
        for item in list_of_assistants:
            print(f"assistant is {item}")

        if "kevin-personal-assistant" not in list_of_assistants:
            print("About to create pinecone assistant")
            self.assistant = pc.assistant.create_assistant(
                assistant_name="kevin-personal-assistant",
                instructions="""You are Kevin Kakolla's AI assistant representing him to recruiters.
                                    Answer questions about his background confidently and accurately based
                                    on his CV and LinkedIn. Keep answers concise since this is a voice call.
                                    Do not use bullet points, markdown, or formatting. If asked about
                                    availability or salary, say Kevin is open to discussing details directly""",
                timeout=30,  # Wait 30 seconds for assistant operation to complete.
            )
        else:
            pc.assistant.delete_assistant(
                assistant_name="kevin-personal-assistant",
            )
            self.assistant = pc.assistant.create_assistant(
                assistant_name="kevin-personal-assistant",
                instructions="""You are Kevin Kakolla's AI assistant representing him to recruiters.
                                    Answer questions about his background confidently and accurately based
                                    on his CV and LinkedIn. Keep answers concise since this is a voice call.
                                    Do not use bullet points, markdown, or formatting. If asked about
                                    availability or salary, say Kevin is open to discussing details directly""",
                timeout=30,  # Wait 30 seconds for assistant operation to complete.
            )

    def ingest_files(self, file, source, document_type):
        response1 = self.assistant.upload_file(
            file_path=file,
            metadata={"source": source, "document_type": document_type},
            timeout=None,
        )

    async def get_query(self, msg: str) -> str:
        response = await asyncio.to_thread(
            self.assistant.chat(messages=[Message(role="user", content=msg)])
        )
        return response.message.content
