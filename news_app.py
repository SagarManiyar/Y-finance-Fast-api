# news_app.py
import os
import requests
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from environment
NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")

class NewsProvider:
    """News provider using NewsData.io API"""

    def __init__(self):
        self.api_key = NEWS_API_KEY

    def expand_company_name(self, ticker: str):
        """Expand ticker into company name using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            name = stock.info.get("longName", None)
            if name:
                return name
        except Exception as e:
            logger.warning(f"Could not expand ticker {ticker}: {str(e)}")
        return None

    def build_query(self, ticker, company_name=None, use_expansion=False):
        """Builds a safe NewsData.io query string"""
        if use_expansion and company_name:
            return f'"{ticker}" OR "{company_name}"'
        return f'"{ticker}"'

    def fetch_news(self, query, max_articles=20, start_date=None, end_date=None):
        """Fetch news from NewsData.io with proper pagination"""
        if not self.api_key:
            st.error("Missing NewsData.io API key. Set it in environment as NEWSDATA_API_KEY.")
            return []

        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": self.api_key,
            "q": query,
            "language": "en"
        }

        # Add date filters if provided
        # if start_date:
        #     params["from_date"] = start_date.strftime('%Y-%m-%d')
        # if end_date:
        #     params["to_date"] = end_date.strftime('%Y-%m-%d')



        results = []
        seen_urls = set()
        next_page = None

        try:
            while len(results) < max_articles:
                # add nextPage token if available
                if next_page:
                    params["page"] = next_page
                elif "page" in params:
                    del params["page"]

                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    st.error(f"Request error: {resp.status_code} {resp.text}")
                    break

                data = resp.json()
                articles = data.get("results", [])

                if not articles:
                    break

                for article in articles:
                    link = article.get("link")
                    if not link or link in seen_urls:
                        continue
                    seen_urls.add(link)

                    results.append({
                        "title": article.get("title", "No title"),
                        "summary": self._better_summary(article.get("description") or article.get("content") or ""),
                        "link": link,
                        "source": article.get("source_id", "Unknown"),
                        "pub_date": article.get("pubDate", "Unknown")
                    })

                    if len(results) >= max_articles:
                        break

                next_page = data.get("nextPage")
                if not next_page:
                    break

        except Exception as e:
            st.error(f"Error fetching news: {e}")
            logger.error(f"News fetch error: {str(e)}")

        return results

    def _better_summary(self, text, sentences=3):
        """Improved summarizer: take first N sentences"""
        if not text:
            return "No summary available"
        parts = text.split(". ")
        return ". ".join(parts[:sentences]) + ("..." if len(parts) > sentences else "")


def display_news_app(ticker: str = None):
    """
    Display the news application interface

    Args:
        ticker: Stock ticker symbol to search for news
    """

    # Page configuration for news
    st.set_page_config(
        page_title=f"Stock News - {ticker}" if ticker else "Stock News Tracker",
        page_icon="📰",
        layout="wide"
    )

    # Initialize news provider
    news_provider = NewsProvider()

    # Header
    st.title("📰 Stock News Tracker")
    st.markdown("Latest news and analysis for stock symbols")

    # Input controls
    col1, col2 = st.columns([3, 1])

    with col1:
        # Use provided ticker or allow manual input
        ticker_input = st.text_input(
            "Enter stock ticker (e.g. AAPL, TSLA):",
            value=ticker.upper() if ticker else "",
            help="Stock ticker symbol to search for news"
        ).strip().upper()

    with col2:
        use_expansion = st.checkbox(
            "Expand with company name",
            value=False,
            help="Include company full name in search query"
        )

    # Date range controls
    st.subheader("📅 Date Range")

    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.selectbox(
            "Select date range:",
            ["1 day", "1 week", "1 month", "3 months", "Custom"],
            index=2  # Default to 1 month
        )

    # Date range calculation
    end_date = datetime.now().date()

    if date_range == "1 day":
        start_date = end_date - timedelta(days=1)
    elif date_range == "1 week":
        start_date = end_date - timedelta(days=7)
    elif date_range == "1 month":
        start_date = end_date - timedelta(days=30)
    elif date_range == "3 months":
        start_date = end_date - timedelta(days=90)
    else:  # Custom
        with col2:
            start_date = st.date_input(
                "Start date",
                value=end_date - timedelta(days=7),
                max_value=end_date
            )
        with col3:
            end_date = st.date_input(
                "End date",
                value=end_date,
                min_value=start_date,
                max_value=datetime.now().date()
            )

    # Additional controls
    col1, col2 = st.columns(2)

    with col1:
        max_articles = st.slider(
            "Maximum articles to fetch",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of articles to retrieve"
        )

    with col2:
        st.write("")  # Spacing
        search_button = st.button("🔍 Get News", type="primary")

    # Main search and display logic
    if search_button or (ticker_input and st.session_state.get('auto_search_news', False)):
        if not ticker_input:
            st.error("Please enter a ticker symbol.")
            return

        # Clear auto search flag
        if 'auto_search_news' in st.session_state:
            del st.session_state['auto_search_news']

        # Get company name if expansion is enabled
        company_name = None
        if use_expansion:
            with st.spinner(f"Getting company information for {ticker_input}..."):
                company_name = news_provider.expand_company_name(ticker_input)
                if company_name:
                    st.info(f"Found company: {company_name}")
                else:
                    st.warning("Could not find company name, searching with ticker only")

        # Build search query
        query = news_provider.build_query(ticker_input, company_name, use_expansion)

        # Fetch news
        with st.spinner(f"🔍 Searching for news with query: {query}"):
            articles = news_provider.fetch_news(
                query=query,
                max_articles=max_articles,
                start_date=start_date,
                end_date=end_date
            )

        # Display results
        if not articles:
            st.warning("No articles found for the specified criteria.")
            st.info("Try adjusting the date range or search parameters.")
        else:
            # Results header
            st.success(f"📰 Found {len(articles)} articles for {ticker_input}")

            if company_name:
                st.info(f"Company: {company_name}")

            st.markdown(f"**Search Query:** {query}")
            st.markdown(f"**Date Range:** {start_date} to {end_date}")

            # Sort articles by publication date (most recent first)
            try:
                # Attempt to sort by date if available
                articles_with_dates = [a for a in articles if a.get('pub_date') and a['pub_date'] != 'Unknown']
                articles_without_dates = [a for a in articles if not a.get('pub_date') or a['pub_date'] == 'Unknown']

                if articles_with_dates:
                    # Simple date sorting - may need improvement based on date format
                    articles_with_dates.sort(key=lambda x: x['pub_date'], reverse=True)
                    articles = articles_with_dates + articles_without_dates
            except Exception as e:
                logger.warning(f"Could not sort articles by date: {str(e)}")

            # Display articles
            st.subheader("📄 News Articles")

            for i, article in enumerate(articles, 1):
                with st.container():
                    # Article header
                    st.markdown(f"### {i}. [{article['title']}]({article['link']})")

                    # Article metadata
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.caption(f"**Source:** {article['source']}")
                    with col2:
                        if article.get('pub_date') and article['pub_date'] != 'Unknown':
                            st.caption(f"**Published:** {article['pub_date']}")
                    with col3:
                        st.caption(f"**#{i}**")

                    # Article summary
                    st.write(article["summary"])

                    # Separator
                    if i < len(articles):
                        st.markdown("---")

            # Download option
            st.subheader("💾 Export Options")

            if st.button("📄 Export to Text"):
                # Create text export
                export_text = f"Stock News Report for {ticker_input}\n"
                export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                export_text += f"Query: {query}\n"
                export_text += f"Date Range: {start_date} to {end_date}\n"
                export_text += f"Articles Found: {len(articles)}\n\n"

                for i, article in enumerate(articles, 1):
                    export_text += f"{i}. {article['title']}\n"
                    export_text += f"Source: {article['source']}\n"
                    export_text += f"Link: {article['link']}\n"
                    if article.get('pub_date') and article['pub_date'] != 'Unknown':
                        export_text += f"Published: {article['pub_date']}\n"
                    export_text += f"Summary: {article['summary']}\n\n"

                st.download_button(
                    label="Download News Report",
                    data=export_text,
                    file_name=f"{ticker_input}_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ News Search Info")

        if ticker_input:
            st.success(f"Current Ticker: **{ticker_input}**")
        else:
            st.info("Enter a ticker symbol to search for news")

        st.markdown("### 🔧 Search Tips:")
        st.markdown("""
        - Use official ticker symbols (e.g., AAPL, GOOGL)
        - Enable company name expansion for broader results
        - Adjust date range for specific time periods
        - Try different article limits for more/fewer results
        """)

        st.markdown("### 📊 Data Source:")
        st.markdown("News data provided by [NewsData.io](https://newsdata.io)")

        if not NEWS_API_KEY:
            st.error("⚠️ NewsData.io API key not configured")
            st.info("Set NEWSDATA_API_KEY environment variable")


def launch_news_tab(ticker: str):
    """
    Launch news application in new tab/page

    Args:
        ticker: Stock ticker symbol
    """
    # Store ticker in session state for news app
    st.session_state['news_ticker'] = ticker
    st.session_state['auto_search_news'] = True

    # Switch to news page
    st.switch_page("news_app.py")


# Main execution for standalone running
if __name__ == "__main__":
    # Check if ticker is provided in session state
    ticker = st.session_state.get('news_ticker', None)
    display_news_app(ticker)