# Throwaway spike server for the Laya evaluation. Not part of the app.
import sys, time
sys.path.insert(0, "model")
from fastapi import FastAPI
from pydantic import BaseModel
from rl_agent_api import RLAgent

GUIDE = {
    "INTEREST_RATE": "policy rate decisions, FOMC/MPC statements, rate outlook, mortgage rates, bond/gilt yields",
    "PROPERTY_MARKET": "house prices, rents, housing supply, construction, mortgage lending volumes",
    "POLICY_CHANGE": "new laws, taxes, or regulations that change what landlords/borrowers pay or can do",
    "ECONOMIC": "inflation, GDP, jobs, economic projections, financial stability",
    "OTHER": "internal bank admin, committee minutes on technical topics, enforcement against individual banks/employees, banknotes, appointments, payments tech, AI, events",
}
RELEVANCE = ["no effect on landlords", "indirect banking-sector news", "economic data that shifts the rate outlook",
             "rate decision or rule that directly changes mortgage costs, rents or property tax"]
BANDS = [0.1, 0.3, 0.6, 0.9]

agent = RLAgent("model", device="cpu")
app = FastAPI()

class Article(BaseModel):
    title: str
    description: str | None = None

@app.post("/decide")
def decide(a: Article):
    t = time.time()
    state = f"Title: {a.title}\nDescription: {a.description or 'N/A'}"
    out = agent.system_one(state, {
        "category": {"type": "choice", "instructions": "Which category is this central-bank or property news for a residential landlord?", "criteria": GUIDE},
        "relevance": {"type": "score", "instructions": "How much does this affect a residential landlord?", "criteria": RELEVANCE},
    })["answers"]
    cat = out["category"]
    band = BANDS[min(3, max(0, round(out["relevance"]["score"])))]
    relevance = min(band, 0.2) if cat["choice"] == "OTHER" else band
    return {"category": cat["choice"], "relevance": relevance, "confidence": cat["confidence"], "ms": int((time.time() - t) * 1000)}
