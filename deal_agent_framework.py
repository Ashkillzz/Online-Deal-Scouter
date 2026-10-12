import logging
import os
import sys
from typing import List
from dotenv import load_dotenv
import numpy as np
from sklearn.manifold import TSNE
import chromadb

from agents.deals import Deal, Opportunity
from agents.planning_agent import PlanningAgent
from database import DealModel, OpportunityModel, close_db_session, get_db_session, init_db

# Category colors for 3D latent space plot
CATEGORIES = [
    'Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics',
    'Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games'
]
COLORS = ['#ff5555', '#50fa7b', '#f1fa8c', '#bd93f9', '#ff79c6', '#8be9fd', '#ffb86c', '#6272a4']

BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [DealScouter] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)


class DealAgentFramework:
    """
    Central pipeline execution controller for Online Deal Scouter.
    Manages database sessions, vector store collections, and planning agent runs.
    """

    DB_PATH = "products_vectorstore"

    def __init__(self):
        init_logging()
        load_dotenv()
        init_db()

        client = chromadb.PersistentClient(path=self.DB_PATH)
        self.collection = client.get_or_create_collection('products')
        self.planner: PlanningAgent | None = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Multi-Agent Framework...")
            self.planner = PlanningAgent(self.collection)
            self.log("Multi-Agent Framework is ready.")

    @property
    def memory(self) -> List[Opportunity]:
        """Retrieve all detected deal opportunities from the database ordered by discount."""
        session = get_db_session()
        try:
            opportunities_db = session.query(OpportunityModel).order_by(
                OpportunityModel.discount.desc()
            ).all()
            opportunities = [
                Opportunity(
                    deal=Deal(
                        product_description=opp.deal.product_description,
                        price=opp.deal.price,
                        url=opp.deal.url
                    ),
                    estimate=opp.estimate,
                    discount=opp.discount
                )
                for opp in opportunities_db if opp.deal
            ]
            return opportunities
        except Exception as e:
            self.log(f"Error fetching opportunities: {e}")
            return []
        finally:
            close_db_session(session)

    def save_opportunity(self, opportunity: Opportunity) -> None:
        """Persist a newly detected opportunity with transaction safety."""
        session = get_db_session()
        try:
            deal_db = session.query(DealModel).filter_by(url=opportunity.deal.url).first()
            if not deal_db:
                deal_db = DealModel(
                    product_description=opportunity.deal.product_description,
                    price=opportunity.deal.price,
                    url=opportunity.deal.url
                )
                session.add(deal_db)
                session.flush()

            opp_db = OpportunityModel(
                deal_id=deal_db.id,
                estimate=opportunity.estimate,
                discount=opportunity.discount
            )
            session.add(opp_db)
            session.commit()
            self.log(f"Saved opportunity with ${opportunity.discount:.2f} discount margin")
        except Exception as e:
            session.rollback()
            self.log(f"Error saving opportunity: {e}")
            raise
        finally:
            close_db_session(session)

    def log(self, message: str):
        text = f"{BG_BLUE}{WHITE}[DealScouter Framework]{RESET} {message}"
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """Executes an autonomous scan and valuation cycle."""
        self.init_agents_as_needed()
        self.log("Kicking off Planning Agent scan cycle")
        memory_urls = [opp.deal.url for opp in self.memory]
        results = self.planner.plan(memory=memory_urls)
        
        self.log(f"Planning Agent cycle finished. Surfaced {len(results)} profitable opportunities.")
        for opp in results:
            self.save_opportunity(opp)
        return self.memory


    @classmethod
    def get_plot_data(cls, max_datapoints: int = 1000):
        """Extracts vector embeddings from ChromaDB and computes 3D t-SNE coordinates."""
        try:
            client = chromadb.PersistentClient(path=cls.DB_PATH)
            collection = client.get_or_create_collection('products')
            result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
            
            if result and result.get('embeddings') and len(result['embeddings']) > 10:
                vectors = np.array(result['embeddings'])
                documents = result['documents']
                categories = [m.get('category', 'Electronics') for m in result['metadatas']]
                colors = [
                    COLORS[CATEGORIES.index(c) % len(COLORS)] if c in CATEGORIES else '#bd93f9'
                    for c in categories
                ]
                perplexity = min(30, max(5, len(vectors) - 1))
                tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
                reduced_vectors = tsne.fit_transform(vectors)
                return documents, reduced_vectors, colors
        except Exception as e:
            logging.info(f"Could not load vector plot data: {e}")

        # Fallback dummy coordinates if vector store is not yet seeded
        n = 50
        fake_vectors = np.random.randn(n, 3)
        fake_docs = [f"Product {i}" for i in range(n)]
        fake_colors = ['#8be9fd'] * n
        return fake_docs, fake_vectors, fake_colors


if __name__ == "__main__":
    DealAgentFramework().run()
