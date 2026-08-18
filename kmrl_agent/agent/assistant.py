import json
import ollama
from kmrl_agent.agent.router import IntentRouter
from kmrl_agent.tools.gtfs_tool import GTFSTool
from kmrl_agent.tools.lost_found_tool import LostFoundTool
from kmrl_agent.tools.vision_tool import VisionTool
from kmrl_agent.tools.kb_search_tool import KBSearchTool

SYSTEM_SYNTHESIZER_PROMPT = """You are the official Agentic RAG AI Assistant for Kochi Metro Rail Limited (KMRL).

Your job is to answer passenger questions about metro routes, stations, fares, timings, lost & found items, policies, and official financial information.

Strict Rules:
1. Use ONLY the retrieved tool context. Do not use outside knowledge or assumptions.
2. Never hallucinate. If the required information is not available in the retrieved context, say: "I could not find this information in the available KMRL sources."
3. Use exact values when provided, including fares, station names, dates, timings, distances, and amounts.
4. Keep answers extremely short and direct. Give only the information needed to answer the question. Do not give extra information.
5. Do not repeat the user's question.
6. Avoid unnecessary explanations, introductions, conclusions, and pleasantries.
7. Use bullets or a short sentence when they make the answer clearer.
8. Always mention the source using a concise citation such as `[GTFS Engine]`, `[Lost & Found Database]`, `[Annual Report PDF]`, or `[Infographic Chart]`.
9. Do not combine information from different sources unless the retrieved context supports the combination.
10. If the retrieved sources conflict, do not guess. Clearly state that the available sources contain conflicting information.

Response Style:
Concise
Factual
Professional
Direct
No unnecessary details

Target: Answer in 1–3 sentences or a few bullets whenever possible.
"""

CHITCHAT_SYSTEM_PROMPT = """You are the official AI Assistant for Kochi Metro Rail Limited (KMRL).
The user has sent a greeting, farewell, thanks, or other casual remark that is not a question about KMRL services.

Reply briefly and warmly in 1-2 sentences. Do not mention sources, citations, or retrieved context.
If it fits naturally, invite them to ask about fares, routes, lost & found, parking, or feeder buses.
"""

class KMRLAssistant:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.router = IntentRouter(model_name=model_name)
        self.gtfs_tool = GTFSTool()
        self.lost_found_tool = LostFoundTool()
        self.vision_tool = VisionTool()
        self.kb_search_tool = KBSearchTool()

    def process_query(self, user_query: str) -> dict:
        # Step 1: Intent Routing
        route_decision = self.router.route(user_query)
        tool_name = route_decision.get("selected_tool", "kb_search")
        params = route_decision.get("parameters", {})
        reasoning = route_decision.get("reasoning", "")

        context_blocks = []
        executed_tools = []

        # Step 2: Tool Execution
        if tool_name == "gtfs":
            executed_tools.append("gtfs_tool")
            orig = params.get("origin")
            dest = params.get("destination")
            stn = params.get("station")
            action = params.get("action")

            if orig and dest:
                res_fare = self.gtfs_tool.get_fare(orig, dest)
                res_route = self.gtfs_tool.find_route(orig, dest)
                context_blocks.append(f"[GTFS Fare Info]: {res_fare.get('formatted')}")
                context_blocks.append(f"[GTFS Route Info]: {res_route.get('formatted')}")
            elif stn and action == "next_trains":
                res_trains = self.gtfs_tool.get_next_trains(stn)
                context_blocks.append(f"[GTFS Train Schedule]: {res_trains.get('formatted')}")
            else:
                # Fallback to general GTFS lookup
                res_fare = self.gtfs_tool.get_fare(user_query, user_query)
                context_blocks.append(f"[GTFS Info]: {json.dumps(res_fare)}")

        elif tool_name == "lost_found":
            executed_tools.append("lost_found_tool")
            lfid = params.get("lfid")
            if lfid:
                res_item = self.lost_found_tool.get_item_by_lfid(lfid)
                context_blocks.append(f"[Lost & Found DB]: {res_item.get('formatted')}")
            else:
                q = params.get("query") or user_query
                stn = params.get("station")
                cat = params.get("category")
                res_search = self.lost_found_tool.search_lost_items(query=q, category=cat, station=stn, limit=5)
                context_blocks.append(f"[Lost & Found DB]: {res_search.get('formatted')}")

        elif tool_name == "vision":
            executed_tools.append("vision_tool")
            res_vis = self.vision_tool.query_visual_data(user_query)
            context_blocks.append(f"[Infographic Vision Data]: {res_vis.get('formatted')}")

        elif tool_name == "chitchat":
            executed_tools.append("chitchat")
            # No tool execution needed - handled directly by the synthesizer below.

        elif tool_name == "multi":
            executed_tools.extend(["lost_found_tool", "kb_search_tool"])
            res_lf = self.lost_found_tool.search_lost_items(query=user_query, limit=3)
            res_kb = self.kb_search_tool.search(query=user_query, top_k=3)
            context_blocks.append(f"[Lost & Found DB]: {res_lf.get('formatted')}")
            context_blocks.append(f"[KMRL Knowledge Base]: {res_kb.get('formatted')}")

        else: # kb_search
            executed_tools.append("kb_search_tool")
            res_kb = self.kb_search_tool.search(query=user_query, top_k=5)
            context_blocks.append(f"[KMRL Knowledge Base]: {res_kb.get('formatted')}")

        assembled_context = "\n\n".join(context_blocks)

        # Step 3: Response Synthesis via Ollama LLM
        if tool_name == "chitchat":
            system_prompt = CHITCHAT_SYSTEM_PROMPT
            prompt = user_query
        else:
            system_prompt = SYSTEM_SYNTHESIZER_PROMPT
            prompt = f"""Retrieved Evidence Context:
{assembled_context}

User Question: {user_query}
Provide a complete, structured answer:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            final_answer = response["message"]["content"]
        except Exception as e:
            final_answer = f"Synthesizer error: {e}\n\nRaw Retrieved Evidence:\n{assembled_context}"

        return {
            "query": user_query,
            "tool_selected": tool_name,
            "reasoning": reasoning,
            "executed_tools": executed_tools,
            "retrieved_context": assembled_context,
            "answer": final_answer
        }

if __name__ == "__main__":
    assistant = KMRLAssistant()
    res = assistant.process_query("How much is ticket fare from Aluva to Pettah?")
    print("TOOL:", res["tool_selected"])
    print("ANSWER:\n", res["answer"])
