import os
import requests
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

# თითოეული მომხმარებლისთვის ცალკე თარო (thread_id-ის მიხედვით)
SHELVES_BY_THREAD = {}


def _get_shelf(thread_id: str) -> list:
    return SHELVES_BY_THREAD.setdefault(thread_id, [])


@tool
def search_books(query: str) -> list:
    """Search books in Open Library by title, author, or topic. Returns top 3 matches."""
    try:
        response = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": 3},
            timeout=15,
        )
        response.raise_for_status()
        docs = response.json().get("docs", [])
        results = []
        for doc in docs:
            results.append({
                "title": doc.get("title", "Unknown"),
                "author": ", ".join(doc.get("author_name", ["Unknown"])),
                "first_publish_year": doc.get("first_publish_year", "Unknown"),
            })
        return results if results else [{"message": "No books found."}]
    except Exception as e:
        return [{"error": f"Open Library unavailable: {e}"}]


@tool
def get_historical_context(topic_or_year: str) -> str:
    """Get a short historical summary from Wikipedia about an author or publication year."""
    headers = {"User-Agent": "LiteraryCuratorAgent/1.0 (Educational Project)"}
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": topic_or_year, "format": "json", "srlimit": 1},
            headers=headers,
            timeout=15,
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia context found for '{topic_or_year}'."
        page_title = hits[0]["title"]
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}",
            headers=headers,
            timeout=15,
        )
        if summary_resp.status_code == 200:
            return f"{page_title}: {summary_resp.json().get('extract', '')[:450]}..."
        return f"Found topic: {page_title}"
    except Exception as e:
        return f"Wikipedia temporarily rate-limited ({e}). Answer briefly using your own historical knowledge."


@tool
def get_weather(city: str = "Tbilisi") -> dict:
    """Get current weather temperature (Celsius) and precipitation for a city."""
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results", [])
        if not results:
            return {"error": f"City '{city}' not found."}
        lat, lon, name = results[0]["latitude"], results[0]["longitude"], results[0]["name"]
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m,precipitation"},
            timeout=15,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current", {})
        return {
            "city": name,
            "temperature_c": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation"),
        }
    except Exception as e:
        return {"error": f"Weather service unavailable: {e}"}


def make_shelf_tools(thread_id: str):
    @tool
    def add_book(title: str, author: str, year: str = "Unknown") -> str:
        """Add a book to the user's personal reading shelf."""
        shelf = _get_shelf(thread_id)
        for b in shelf:
            if b["title"].lower() == title.lower():
                return f"'{title}' is already on your shelf."
        shelf.append({"title": title, "author": author, "year": str(year), "notes": []})
        return f"Added '{title}' by {author} ({year}) to your shelf."

    @tool
    def list_books() -> list:
        """List all books currently on the user's reading shelf."""
        shelf = _get_shelf(thread_id)
        return shelf if shelf else [{"message": "Your reading shelf is empty."}]

    @tool
    def log_reading_note(title: str, note: str) -> str:
        """Save a reading note or quote for a book on the user's shelf."""
        shelf = _get_shelf(thread_id)
        for b in shelf:
            if b["title"].lower() == title.lower():
                b["notes"].append(note)
                return f"Saved note for '{b['title']}'."
        return f"Book '{title}' was not found on your shelf."

    @tool
    def get_reading_notes(title: str) -> str:
        """Retrieve saved notes for a book on the user's shelf."""
        shelf = _get_shelf(thread_id)
        for b in shelf:
            if b["title"].lower() == title.lower():
                if not b["notes"]:
                    return f"No notes saved yet for '{b['title']}'."
                return f"Notes for '{b['title']}': " + " | ".join(b["notes"])
        return f"Book '{title}' was not found on your shelf."

    @tool
    def delete_book(title: str) -> str:
        """Delete a book from the user's reading shelf (requires human confirmation)."""
        shelf = _get_shelf(thread_id)
        target = next((b for b in shelf if b["title"].lower() == title.lower()), None)
        if not target:
            return f"Book '{title}' is not on your shelf."
        decision = interrupt(
            f"Are you sure you want to delete '{target['title']}' by {target['author']} from your shelf?"
        )
        if str(decision).strip().lower() in ("yes", "y", "true", "confirm", "დიახ"):
            shelf.remove(target)
            return f"Deleted '{target['title']}' from your shelf."
        return f"Deletion of '{target['title']}' was cancelled."

    return [add_book, list_books, log_reading_note, get_reading_notes, delete_book]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are the 'Literary Time-Machine & Reading Club Concierge', a warm and knowledgeable literary assistant.
Use Open Library (`search_books`), Wikipedia (`get_historical_context`), Open-Meteo (`get_weather`), and shelf tools (`add_book`, `list_books`, `log_reading_note`, `get_reading_notes`, `delete_book`).
Always call `delete_book` when asked to delete a book. Respond clearly and concisely in the user's language (Georgian or English)."""


def build_graph(thread_id: str = "default"):
    tools = [search_books, get_historical_context, get_weather] + make_shelf_tools(thread_id)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, frequency_penalty=0.6, max_tokens=1024)
    llm_with_tools = llm.bind_tools(tools)

    def assistant_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=InMemorySaver())
