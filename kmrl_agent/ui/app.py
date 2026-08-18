import sys
import os
import streamlit as st

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kmrl_agent.agent.assistant import KMRLAssistant
from kmrl_agent.tools.gtfs_tool import GTFSTool
from kmrl_agent.tools.lost_found_tool import LostFoundTool

# Page Configuration
st.set_page_config(
    page_title="Kochi Metro (KMRL) AI Assistant",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #008080;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #f0f4f8;
        border-left: 4px solid #008080;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assistant():
    return KMRLAssistant(model_name="llama3.1:8b")

@st.cache_resource
def load_gtfs():
    return GTFSTool()

@st.cache_resource
def load_lost_found():
    return LostFoundTool()

assistant = load_assistant()
gtfs_tool = load_gtfs()
lf_tool = load_lost_found()

st.markdown('<div class="main-title">🚇 Kochi Metro (KMRL) Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">100% Open-Source Multi-Tool RAG • GTFS Transit Engine • Lost & Found Lookup • Annual Reports</div>', unsafe_allow_html=True)

# Sidebar Widgets
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/eb/Kochi_Metro_logo.svg", width=160)
    st.header("⚡ Quick Utility Tools")

    # Widget 1: GTFS Fare & Route Calculator
    with st.expander("🎫 Journey & Fare Calculator", expanded=True):
        stations_data = gtfs_tool.get_all_stations()
        station_names = [s["stop_name"] for s in stations_data] if stations_data else ["Aluva", "Pettah", "Edapally", "MG Road"]

        orig_stn = st.selectbox("Origin Station", station_names, index=0)
        dest_stn = st.selectbox("Destination Station", station_names, index=min(len(station_names)-1, 15))

        if st.button("Calculate Fare & Route", use_container_width=True):
            route_res = gtfs_tool.find_route(orig_stn, dest_stn)
            if "error" not in route_res:
                st.success(f"**Fare:** Rs. {route_res['fare_price']}")
                st.info(f"**Stops:** {route_res['num_stops']} | **Est. Time:** {route_res['estimated_minutes']} mins")
                st.write("**Route:** " + " ➔ ".join(route_res["intermediate_stations"]))
            else:
                st.error(route_res["error"])

    # Widget 2: Lost & Found Search Widget
    with st.expander("🔍 Lost Item Lookup"):
        lf_query = st.text_input("Item Name or LFID (e.g. Umbrella, ALVA...)", key="sidebar_lf")
        if st.button("Search Lost Database", use_container_width=True):
            if lf_query:
                lf_res = lf_tool.search_lost_items(query=lf_query, limit=3)
                if lf_res["count"] > 0:
                    for item in lf_res["items"]:
                        st.markdown(f"**[{item['lfid']}]** {item['category']}\n- Date: {item['found_date']} | Found At: {item['found_station']} | Current: {item['current_station']}")
                else:
                    st.warning("No records found matching criteria.")

    st.markdown("---")
    st.markdown("**Powered by:** Ollama (`llama3.1:8b`, `nomic-embed-text`), SQLite, BM25, Streamlit.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Kochi Metro Rail Limited (KMRL)! How can I help you today? Ask me about train fares, routes, lost & found items, parking rates, feeder bus timings, or official policies."}
    ]

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "debug_info" in msg:
            with st.expander("🛠️ Agent Execution Transparency"):
                st.write("**Selected Tool:**", msg["debug_info"].get("tool_selected"))
                st.write("**Reasoning:**", msg["debug_info"].get("reasoning"))
                st.write("**Executed Modules:**", msg["debug_info"].get("executed_tools"))
                st.text_area("Retrieved Tool Context", msg["debug_info"].get("retrieved_context"), height=150, key=f"debug_ctx_{idx}")

# User Chat Input
if user_input := st.chat_input("Ask about fares, train timings, lost items, parking..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent routing & gathering context from specialized engines..."):
            res = assistant.process_query(user_input)
            st.markdown(res["answer"])

            with st.expander("🛠️ Agent Execution Transparency"):
                st.write("**Selected Tool:**", res["tool_selected"])
                st.write("**Reasoning:**", res["reasoning"])
                st.write("**Executed Modules:**", res["executed_tools"])
                st.text_area("Retrieved Tool Context", res["retrieved_context"], height=150, key=f"debug_ctx_{len(st.session_state.messages)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "debug_info": {
                    "tool_selected": res["tool_selected"],
                    "reasoning": res["reasoning"],
                    "executed_tools": res["executed_tools"],
                    "retrieved_context": res["retrieved_context"]
                }
            })
