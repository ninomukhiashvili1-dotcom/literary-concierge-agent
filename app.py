import os
import uuid
import streamlit as st
from langgraph.types import Command
from agent import build_graph

st.set_page_config(page_title="Literary Time-Machine Concierge", page_icon="📚")
st.title("📚 Literary Time-Machine & Reading Club Concierge")
st.caption("აღმოაჩინე წიგნები, გაიგე მათი გამოცემის ეპოქის ისტორია, შეამოწმე ამინდი საკითხავი ადგილისთვის და მართე პირადი თარო.")

if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(st.session_state.thread_id)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

if st.session_state.pending_interrupt:
    st.warning(f"⚠️ დადასტურება: {st.session_state.pending_interrupt}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ დადასტურება (წაშლა)", use_container_width=True):
            try:
                res = st.session_state.graph.invoke(Command(resume=True), config)
                reply = res["messages"][-1].content
                st.session_state.chat_history.append(("assistant", reply))
                st.session_state.pending_interrupt = None
                st.rerun()
            except Exception as e:
                st.error(f"შეცდომა API-სთან კავშირისას: {e}")
    with col2:
        if st.button("❌ უარი (გაუქმება)", use_container_width=True):
            try:
                res = st.session_state.graph.invoke(Command(resume=False), config)
                reply = res["messages"][-1].content
                st.session_state.chat_history.append(("assistant", reply))
                st.session_state.pending_interrupt = None
                st.rerun()
            except Exception as e:
                st.error(f"შეცდომა API-სთან კავშირისას: {e}")

prompt = st.chat_input("მკითხე წიგნებზე, ისტორიაზე, ამინდზე ან მართე თარო...")
if prompt and not st.session_state.pending_interrupt:
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        res = st.session_state.graph.invoke({"messages": [("user", prompt)]}, config)
        state = st.session_state.graph.get_state(config)
        if state.next:
            interrupt_msg = state.tasks[0].interrupts[0].value
            st.session_state.pending_interrupt = interrupt_msg
            st.rerun()
        else:
            reply = res["messages"][-1].content
            st.session_state.chat_history.append(("assistant", reply))
            with st.chat_message("assistant"):
                st.markdown(reply)
    except Exception as e:
        st.error(f"API დროებით მიუწვდომელია ან ლიმიტი ამოიწურა: {e}")
