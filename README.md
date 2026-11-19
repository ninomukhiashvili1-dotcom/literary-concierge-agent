# Literary Time-Machine & Reading Club Concierge

An AI-powered reading assistant built with **LangGraph**, **ChatGroq**, and **Streamlit**.

## Features & External APIs
1. **Open Library API (`search_books`):** Discovers books by title, author, or topic with publication years.
2. **Wikipedia API (`get_historical_context`):** Provides historical context about the author or the book's publication year.
3. **Open-Meteo API (`get_weather`):** Checks real-time weather to recommend indoor or outdoor reading spots.
4. **Personal Reading Shelf (`add_book`, `list_books`, `log_reading_note`, `get_reading_stats`, `delete_book`):** Stateful shelf tracking with human-in-the-loop approval (`interrupt()`) before book deletion.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
