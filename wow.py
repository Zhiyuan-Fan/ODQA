import rocketqa
from docarray import Document,DocumentArray
from jina import Executor,requests,Flow
import os
from multiprocessing import set_start_method as _set_start_method
import numpy as np
import sys
from pathlib import Path
import time



os.environ.setdefault('JINA_USE_CUDA', 'True')

class RocketqaDeExecutor(Executor):
    def __init__(self,model_name="zh_dureader_de",use_Cuda=True,device_Id=0,batch_Size=32,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = rocketqa.load_model(model=model_name,use_cuda=True,device_id=device_Id,batch_size=batch_Size)

    @requests(on="/index")
    def encode_passage(self,docs:DocumentArray,**kwargs):

        embeddings = self.model.encode_para(para=docs.texts)
        """print(embeddings)
        print("down")"""
        """for embedding in embeddings:
            print(embedding)"""
        docs.embeddings = [embedding for embedding in embeddings]
        #docs.embeddings = embeddings

        
    
    @requests(on="/search")
    def encode_query(self,docs,**kwargs):
        print("retriever is working......")
        start = time.time()
        for doc in  docs:
            #queryListStr = queryDocument.content
          
            generator_temp = self.model.encode_query(query=[doc.text])
            for temp in generator_temp:
                doc.embedding = temp
            #print(doc.embedding)
        end = time.time()
        print("retrieve time: ",end-start,"s")




class RocketqaCeExecutor(Executor):
    def __init__(self,model_Name="zh_dureader_ce",use_Cuda=True,device_Id=0,batch_Size=32,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = rocketqa.load_model(model=model_Name,use_cuda=True,device_id=device_Id,batch_size=batch_Size)

    @requests(on="/search")
    def rerank(self,docs,**kwargs):
        print("reranker is working......")
        print("召回结果排序中......")
        start = time.time()
        for doc in docs:
            """print("doc text is")
            print(doc.text)"""
            
            str_list = []
            for m in doc.matches:
                #print(m.text)
                #print("score is:")
                """ss = self.model.matching(query=[doc.text],para=[m.text])
                for sco in ss:
                    print(sco)"""
                str_list.append(m.text)
            doc.matches = []
            scores = []
            score_generator = self.model.matching(query=[doc.text]*len(str_list),para=str_list)
            for g in score_generator:
                scores.append(g)

            

            scores = np.array(scores).argsort()
            doc.matches.append(Document(text=str_list[scores[-1]]))
            doc.matches.append(Document(text=str_list[scores[-2]]))
            doc.matches.append(Document(text=str_list[scores[-3]]))
        end = time.time()
        print("rerank time:",end-start,"s")



        


class Indexer(Executor):
    _docs = DocumentArray()  # for storing all documents in memory

    @requests(on='/index')
    def foo(self, docs: DocumentArray, **kwargs):
        print("it is ok")
        self._docs.extend(docs)  # extend stored `docs`

    @requests(on='/search')
    def bar(self, docs: DocumentArray, **kwargs):
        print("all is well")
        docs.match(self._docs, metric='euclidean', limit=20)


test_flow = (
    Flow()
    .add(
        name = "test",
        uses=RocketqaDeExecutor
    )
    .add(
         uses="jinahub://SimpleIndexer/v0.15", install_requirements=True, name="indexer"
    )
    .add(
        uses=RocketqaCeExecutor,
        name="rerank"
    )
)

trydata = Document()
trydata.text = "123"


docs = DocumentArray(
    [
        trydata
    ]
)




def get_passages_from_tsv(file_name):

    pass

def read_file(file_name):
    lines = 0
    with open(file_name) as f:
        for ln,line in enumerate(f):
            """print("number is",ln)
            print(line)"""
            one,two,three,four = line.strip().split('\t')
            """print("one is",one)
            print("two is",two)
            print("three is",three)
            print("four is",four)"""
            doc = Document(
                text = three
            )
            lines = lines + 1
            yield doc
            #print("three is",three)

def main(order):
    if order == 'index':
        if Path('./workspace').exists():
            print('./workspace exists, please deleted it if you want to reindexi')
            return 0
        data_path = sys.argv[2]
        if data_path is None:
            print("No data_path!")
        index(data_path)
    elif order == 'query':
        query()

def index(path):
    with test_flow:
        test_flow.index(inputs=read_file(path), show_progress=True)

def query():
    with test_flow:

        while(True):
            query = input("请输入查询选项：")
            if query == "exit":
                break
            query = Document(text=query)
            docs = test_flow.search(inputs=query)
                
            matches = docs[0].matches
            print("搜索答案为：")
            ids = 1
            for match in matches:
                print("推荐答案排行，NO.",ids)
                print(match.text)
                ids = ids + 1
        

    




"""with f:
    f.post('/index', (Document(text=t.strip()) for t in open('fuzzy-grep.ipynb') if t.strip()))"""

if __name__ == "__main__":
    _set_start_method('spawn')
    order = sys.argv[1]
    main(order)
    


    

def calculate():
    #multi_process
    pass

        
