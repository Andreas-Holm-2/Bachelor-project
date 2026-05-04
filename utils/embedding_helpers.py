# not used in the final project

# import os
# import numpy as np
# from openai import OpenAI
# from dotenv import load_dotenv
# from tqdm import tqdm


# def embed_texts(texts, batch_size=32, MAX_CHARS=1500, MODEL_NAME="Nomic Embed Text"):
#     load_dotenv()
#     client = OpenAI(
#         # api_key=os.getenv("CAMPUSAI_API_KEY"),
#         api_key=os.getenv("CAMPUSAI_API_KEY"),
#         base_url="https://api.campusai.compute.dtu.dk",
#     )
#     truncated = [t[:MAX_CHARS] for t in texts]
#     all_embeddings = []
#     for i in tqdm(range(0, len(truncated), batch_size)):
#         batch = truncated[i : i + batch_size]
#         response = client.embeddings.create(model=MODEL_NAME, input=batch)
#         sorted_data = sorted(response.data, key=lambda e: e.index)
#         vecs = np.array([e.embedding for e in sorted_data])
#         norms = np.linalg.norm(vecs, axis=1, keepdims=True)
#         all_embeddings.append(vecs / np.where(norms == 0, 1, norms))
#     return np.vstack(all_embeddings)
