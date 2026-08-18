import json
import ollama

ROUTER_PROMPT = """You are the Intent Router for the Kochi Metro (KMRL) Agentic AI Assistant.
Analyze the user's input query and determine the correct tool(s) to execute.

Available Tools:
1. "gtfs": Use for train routes, station stops, station-to-station ticket fares, travel times, and scheduled upcoming train times.
   - Parameters to extract: "origin", "destination", "station", "action" ("fare", "route", "next_trains")
2. "lost_found": Use for lost or found items, checking if an item (bag, umbrella, jewelry, phone, wallet, etc.) was found at a station, or looking up a specific LFID.
   - Parameters to extract: "query", "category", "station", "lfid"
3. "vision": Use for parking charges, parking rates, feeder bus timetables/schedules, or network system maps.
   - Parameters to extract: "query"
4. "kb_search": Use for annual report financial data, profit/loss, ridership records, rules, policies, tenders, vigilance, or general inquiry.
   - Parameters to extract: "query"
5. "multi": Use when the query needs BOTH lost item database search AND policy/grievance contact info.
6. "chitchat": Use for greetings ("hi", "hello"), farewells ("bye", "goodbye"), thanks, or casual conversation that is not a question about KMRL services. Nothing to extract, is not a real station name, item, or topic.
   - Parameters: none required.

Respond strictly with a JSON object matching this schema:
{
  "selected_tool": "gtfs" | "lost_found" | "vision" | "kb_search" | "multi" | "chitchat",
  "reasoning": "Brief explanation of tool selection",
  "parameters": { ... extracted tool parameters ... }
}
"""

class IntentRouter:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name

    def route(self, user_query: str) -> dict:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                format="json"
            )
            content = response["message"]["content"]
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"Router fallback due to error: {e}")
            # Fallback heuristic router
            query_lower = user_query.lower().strip(" .!?")
            greeting_words = {"hi", "hii", "hii!", "hello", "hey", "bye", "goodbye", "bye bye",
                               "thanks", "thank you", "thnx", "ty", "good morning", "good evening",
                               "good afternoon", "good night", "how are you"}
            if query_lower in greeting_words or any(query_lower.startswith(g) for g in ["hi ", "hii ", "hello ", "hey ", "thanks", "thank you"]):
                return {
                    "selected_tool": "chitchat",
                    "reasoning": "Fallback greeting/small-talk match",
                    "parameters": {}
                }
            elif any(w in query_lower for w in ["fare", "ticket", "cost", "price", "route", "train", "station"]):
                return {
                    "selected_tool": "gtfs",
                    "reasoning": "Fallback GTFS keyword match",
                    "parameters": {"query": user_query}
                }
            elif any(w in query_lower for w in ["lost", "found", "umbrella", "bag", "wallet", "lfid"]):
                return {
                    "selected_tool": "lost_found",
                    "reasoning": "Fallback Lost & Found keyword match",
                    "parameters": {"query": user_query}
                }
            elif any(w in query_lower for w in ["parking", "feeder bus", "bus schedule", "system map"]):
                return {
                    "selected_tool": "vision",
                    "reasoning": "Fallback Vision keyword match",
                    "parameters": {"query": user_query}
                }
            else:
                return {
                    "selected_tool": "kb_search",
                    "reasoning": "Fallback KB Search match",
                    "parameters": {"query": user_query}
                }

if __name__ == "__main__":
    router = IntentRouter()
    print(router.route("How much is ticket from Aluva to Pettah?"))
    print(router.route("Did anyone find an umbrella at Edapally?"))
    print(router.route("What are the parking charges for two wheelers?"))
    print(router.route("hii"))
