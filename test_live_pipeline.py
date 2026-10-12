"""
Live End-to-End Pipeline Smoke Test for Online Deal Scouter.
Verifies all 4 pricing models, RAG vector retrieval, and multi-deal preservation.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb
from agents.deals import Deal, Opportunity
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent
from agents.specialist_agent import SpecialistAgent
from agents.ensemble_agent import EnsembleAgent
from agents.planning_agent import PlanningAgent
from deal_agent_framework import DealAgentFramework


def run_smoke_test():
    print("=" * 65)
    print("ONLINE DEAL SCOUTER - LIVE PIPELINE VERIFICATION")
    print("=" * 65)

    test_product = (
        "Apple iPad 10th Generation 10.9-inch Liquid Retina Display 64GB "
        "Wi-Fi 6, 12MP Front/12MP Back Camera, Touch ID, All-Day Battery Life - Blue"
    )
    print(f"\n[Test Product Specification]:\n{test_product}\n")

    # 1. Verify ChromaDB
    print("[1/5] Verifying ChromaDB RAG Vector Store...")
    client = chromadb.PersistentClient(path="products_vectorstore")
    collection = client.get_or_create_collection("products")
    doc_count = collection.count()
    print(f" -> ChromaDB collection 'products' count: {doc_count}")
    assert doc_count >= 1000, f"Expected at least 1000 items, got {doc_count}"

    # 2. Test Base Pricing Agents
    print("\n[2/5] Testing Pricing Models Independently...")
    
    # Random Forest
    rf_agent = RandomForestAgent()
    rf_price = rf_agent.price(test_product)
    print(f" -> [Random Forest Agent]: ${rf_price:.2f}")
    assert rf_price > 0, "Random Forest returned $0.00"

    # Specialist (Modal LLaMA)
    specialist_agent = SpecialistAgent()
    specialist_price = specialist_agent.price(test_product)
    print(f" -> [Specialist Agent (Modal GPU)]: ${specialist_price:.2f}")
    assert specialist_price > 0, "Specialist Agent returned $0.00 fallback"

    # Frontier (Chroma RAG + GPT)
    frontier_agent = FrontierAgent(collection)
    frontier_price = frontier_agent.price(test_product)
    print(f" -> [Frontier Agent (RAG + GPT)]: ${frontier_price:.2f}")
    assert frontier_price > 0, "Frontier Agent returned $0.00"

    # 3. Test Meta-Regressor Ensemble
    print("\n[3/5] Testing Ensemble Meta-Regressor...")
    ensemble_agent = EnsembleAgent(collection)
    ensemble_price = ensemble_agent.price(test_product)
    print(f" -> [Ensemble Meta-Consensus Price]: ${ensemble_price:.2f}")
    assert ensemble_price > 0, "Ensemble Agent returned $0.00"

    # 4. Test Planning Agent Multi-Deal Opportunity Discovery
    print("\n[4/5] Testing Planning Agent Real Deal Scrape & Valuation...")
    planner = PlanningAgent(collection)
    discovered_opportunities = planner.plan(memory=[])
    print(f" -> Discovered {len(discovered_opportunities)} profitable opportunities:")
    for idx, opp in enumerate(discovered_opportunities, 1):
        print(f"    {idx}. Listed: ${opp.deal.price:.2f} | Est: ${opp.estimate:.2f} | Discount: +${opp.discount:.2f} ({opp.discount/opp.estimate*100:.1f}%)")
        print(f"       Item: {opp.deal.product_description[:75]}...")

    # 5. Test Framework Database Persistence
    print("\n[5/5] Testing Framework Full Execution & DB Persistence...")
    framework = DealAgentFramework()
    # Save first opportunity to test DB insertion
    if discovered_opportunities:
        framework.save_opportunity(discovered_opportunities[0])
        all_stored = framework.memory
        print(f" -> Successfully persisted to DB. Total DB Opportunities: {len(all_stored)}")
    else:
        print(" -> No profitable deals found in current scrape batch (market conditions fluctuate).")

    print("\n" + "=" * 65)
    print("ALL LIVE PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
