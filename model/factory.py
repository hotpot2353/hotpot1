import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf

#定义一个名叫“基础模型工厂”的抽象类，它继承自 ABC。这意味着这个类不能被直接实例化（不能 BaseModelFactory()），只能被继承。
class BaseModelFactory(ABC):
    @abstractmethod #装饰器，声明下面的方法是一个“抽象方法”。子类必须重写（实现）这个方法，否则子类也无法被实例化。
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):  #定义一个“聊天模型工厂”，它继承了上面的抽象基类
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
