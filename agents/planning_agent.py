from typing import List, Optional

from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.scanner_agent import ScannerAgent


class PlanningAgent(Agent):
    """
    Planning Agent serves as the master workflow coordinator:
    1. Scans and filters deals via ScannerAgent.
    2. Prices top candidates via EnsembleAgent.
    3. Triggers notifications via MessagingAgent for discounts exceeding DEAL_THRESHOLD.
    """

    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50.0  # Alert threshold in USD

    def __init__(self, collection):
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """Evaluate and price a candidate deal."""
        self.log(f"Pricing candidate deal: {deal.product_description[:40]}...")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Calculated discount: ${discount:.2f} (Listed: ${deal.price:.2f}, Est: ${estimate:.2f})")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> List[Opportunity]:
        """Execute the end-to-end deal hunting workflow and return all qualifying opportunities."""
        self.log("Planning Agent starting autonomous scan cycle")
        selection = self.scanner.scan(memory=memory)
        
        qualifying_opportunities: List[Opportunity] = []
        if selection and selection.deals:
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            
            # Select all deals with a positive discount margin
            qualifying_opportunities = [opp for opp in opportunities if opp.discount > 0]
            
            best_opportunity = opportunities[0] if opportunities else None
            if best_opportunity and best_opportunity.discount > self.DEAL_THRESHOLD:
                self.log(f"Top deal discount (${best_opportunity.discount:.2f}) exceeds ${self.DEAL_THRESHOLD:.2f} threshold; dispatching alert")
                self.messenger.alert(best_opportunity)
            elif best_opportunity:
                self.log(f"Top deal discount (${best_opportunity.discount:.2f}) below ${self.DEAL_THRESHOLD:.2f} alert threshold")
                
            self.log(f"Planning cycle complete: {len(qualifying_opportunities)} profitable deals surfaced")
            return qualifying_opportunities
                
        self.log("No new viable deals found this cycle")
        return []

