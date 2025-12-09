"""
Parliamentary Written Questions Search Application

A Streamlit app for searching UK Parliamentary Written Questions
using boolean search with the eldar-extended library.
"""

import html
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Try to import eldar_extended first, fall back to eldar
USING_EXTENDED = False
SEARCH_QUERY_AVAILABLE = False
try:
    from eldar_extended import Query, SearchQuery

    ELDAR_AVAILABLE = True
    USING_EXTENDED = True
    SEARCH_QUERY_AVAILABLE = True
except ImportError:
    try:
        from eldar_extended import Query

        ELDAR_AVAILABLE = True
        USING_EXTENDED = True
    except ImportError:
        try:
            from eldar import Query

            ELDAR_AVAILABLE = True
        except ImportError:
            ELDAR_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "https://questions-statements-api.parliament.uk"
QUESTIONS_ENDPOINT = "/api/writtenquestions/questions"

# Rate limiting configuration
REQUEST_DELAY_LIST = 0.3  # Delay between list pagination requests

# Parallel fetching configuration
PARALLEL_WORKERS = 5  # Number of concurrent requests for detail fetching
CHUNK_SIZE = 50  # Number of questions to fetch per chunk
CHUNK_DELAY = 0.5  # Seconds to pause between chunks
RATE_LIMIT_BACKOFF = 5.0  # Seconds to pause when rate limited
REQUEST_DELAY_DETAIL = 0.05  # Small delay between individual detail requests

# Pagination
DEFAULT_PAGE_SIZE = 100
MAX_RESULTS = 5000  # Safety limit
MAX_DATE_RANGE_DAYS = 5  # Maximum date range allowed

# Highlight styling
HIGHLIGHT_COLOR = "#FFEB3B"  # Yellow highlight
HIGHLIGHT_STYLE = (
    f"background-color: {HIGHLIGHT_COLOR}; padding: 0 2px; border-radius: 2px;"
)

# Party name abbreviations (matching the Chrome extension)
PARTY_ABBREVIATIONS = {
    "Conservative": "Con",
    "Labour": "Lab",
    "Independent": "Ind",
    "Liberal Democrat": "Lib Dem",
    "Liberal Democrats": "Lib Dem",
    "Scottish National Party": "SNP",
    "Democratic Unionist Party": "DUP",
    "Social Democratic & Labour Party": "SDLP",
    "Ulster Unionist Party": "UUP",
    "Green Party": "Green",
    "Plaid Cymru": "PC",
    "Alba Party": "Alba",
    "Reform UK": "Reform",
}


# =============================================================================
# API Client
# =============================================================================


class ParliamentaryQuestionsAPI:
    """Client for the UK Parliament Written Questions API."""

    def __init__(self):
        self.base_url = API_BASE_URL
        # Create a session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=PARALLEL_WORKERS + 2,
            pool_maxsize=PARALLEL_WORKERS + 2,
        )
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "ParliamentaryQuestionsSearch/1.0",
            }
        )

    def fetch_questions_list(
        self,
        date_from: datetime,
        date_to: datetime,
        date_type: str = "tabled",
        house: Optional[str] = None,
        answered: Optional[str] = None,
        skip: int = 0,
        take: int = DEFAULT_PAGE_SIZE,
        expand_member: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch list of written questions from the API (truncated text).
        """
        params = {
            "skip": skip,
            "take": take,
            "expandMember": str(expand_member).lower(),
        }

        # Set date parameters based on date type
        if date_type == "answered":
            params["answeredWhenFrom"] = date_from.strftime("%Y-%m-%d")
            params["answeredWhenTo"] = date_to.strftime("%Y-%m-%d")
        else:  # tabled
            params["tabledWhenFrom"] = date_from.strftime("%Y-%m-%d")
            params["tabledWhenTo"] = date_to.strftime("%Y-%m-%d")

        if house and house != "Both":
            params["house"] = house

        if answered and answered != "All":
            params["answered"] = answered

        try:
            response = self.session.get(
                f"{self.base_url}{QUESTIONS_ENDPOINT}", params=params, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"totalResults": 0, "results": [], "error": str(e)}

    def fetch_question_detail(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch full details for a single question (complete text).
        Thread-safe method for parallel fetching.
        """
        try:
            response = self.session.get(
                f"{self.base_url}{QUESTIONS_ENDPOINT}/{question_id}",
                params={"expandMember": "true"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if "value" in data:
                return data["value"]
            return data
        except requests.exceptions.RequestException:
            return None

    def _estimate_fetch_time(self, num_questions: int) -> str:
        """Estimate time to fetch questions based on chunked approach."""
        num_chunks = (num_questions + CHUNK_SIZE - 1) // CHUNK_SIZE
        # Estimate: each chunk takes ~1-1.5 seconds (parallel fetch) + delay between chunks
        estimated_seconds = num_chunks * (1.5 + CHUNK_DELAY)

        if estimated_seconds < 60:
            return f"~{int(estimated_seconds)} seconds"
        else:
            minutes = int(estimated_seconds // 60)
            seconds = int(estimated_seconds % 60)
            return f"~{minutes}m {seconds}s"

    def fetch_all_questions(
        self,
        date_from: datetime,
        date_to: datetime,
        date_type: str = "tabled",
        house: Optional[str] = None,
        answered: Optional[str] = None,
        max_results: int = MAX_RESULTS,
        progress_bar=None,
        status_text=None,
        stop_flag=None,
    ) -> Dict[str, Any]:
        """
        Fetch all questions within a date range, handling pagination.
        Uses chunked parallel fetching with rate limiting for reliability.

        Returns dict with 'questions', 'total_reported', 'total_fetched', 'errors', 'cancelled'
        """
        question_ids = []
        errors = []
        skip = 0

        # Phase 1: Get list of question IDs (sequential)
        if status_text:
            status_text.text("Phase 1/2: Getting question list...")

        # First request to get total count
        initial_response = self.fetch_questions_list(
            date_from,
            date_to,
            date_type,
            house,
            answered,
            skip=0,
            take=DEFAULT_PAGE_SIZE,
        )

        if "error" in initial_response:
            errors.append(f"Initial request: {initial_response['error']}")
            return {
                "questions": [],
                "total_reported": 0,
                "total_fetched": 0,
                "errors": errors,
                "cancelled": False,
            }

        total_results = initial_response.get("totalResults", 0)
        results_to_fetch = min(total_results, max_results)

        if results_to_fetch == 0:
            return {
                "questions": [],
                "total_reported": total_results,
                "total_fetched": 0,
                "errors": errors,
                "cancelled": False,
            }

        # Process first batch - extract IDs
        for item in initial_response.get("results", []):
            if "value" in item and "id" in item["value"]:
                question_ids.append(item["value"]["id"])

        skip = len(initial_response.get("results", []))

        # Fetch remaining pages to get all IDs
        while skip < results_to_fetch:
            if stop_flag and stop_flag.is_set():
                return {
                    "questions": [],
                    "total_reported": total_results,
                    "total_fetched": 0,
                    "errors": errors,
                    "cancelled": True,
                }

            if progress_bar:
                progress_bar.progress(min((skip / results_to_fetch) * 0.15, 0.15))

            if status_text:
                status_text.text(
                    f"Phase 1/2: Getting question list... {len(question_ids)} / {results_to_fetch}"
                )

            time.sleep(REQUEST_DELAY_LIST)

            response = self.fetch_questions_list(
                date_from,
                date_to,
                date_type,
                house,
                answered,
                skip=skip,
                take=DEFAULT_PAGE_SIZE,
            )

            if "error" in response:
                errors.append(f"List page {skip}: {response['error']}")
                skip += DEFAULT_PAGE_SIZE
                continue

            results = response.get("results", [])
            if not results:
                break

            for item in results:
                if "value" in item and "id" in item["value"]:
                    question_ids.append(item["value"]["id"])

            skip += len(results)

        # Phase 2: Fetch full details in chunks with rate limiting
        total_questions = len(question_ids)
        estimated_time = self._estimate_fetch_time(total_questions)

        if status_text:
            status_text.text(
                f"Phase 2/2: Fetching {total_questions} questions ({estimated_time})..."
            )

        all_questions = []
        rate_limited = False
        consecutive_failures = 0

        # Split into chunks
        chunks = [
            question_ids[i : i + CHUNK_SIZE]
            for i in range(0, total_questions, CHUNK_SIZE)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            if stop_flag and stop_flag.is_set():
                return {
                    "questions": all_questions,
                    "total_reported": total_results,
                    "total_fetched": len(all_questions),
                    "errors": errors,
                    "cancelled": True,
                }

            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + len(chunk), total_questions)

            if status_text:
                status_msg = f"Phase 2/2: Fetching questions {chunk_start + 1}-{chunk_end} of {total_questions}"
                if rate_limited:
                    status_msg += " (slowed down)"
                status_text.text(status_msg)

            # Fetch this chunk in parallel
            chunk_results = []
            chunk_failures = 0

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                # Submit all tasks for this chunk
                futures = {
                    executor.submit(self.fetch_question_detail, qid): qid
                    for qid in chunk
                }

                # Collect results
                for future in as_completed(futures):
                    if stop_flag and stop_flag.is_set():
                        for f in futures:
                            f.cancel()
                        return {
                            "questions": all_questions,
                            "total_reported": total_results,
                            "total_fetched": len(all_questions),
                            "errors": errors,
                            "cancelled": True,
                        }

                    qid = futures[future]
                    try:
                        result = future.result()
                        if result:
                            chunk_results.append(result)
                            consecutive_failures = 0
                        else:
                            chunk_failures += 1
                            consecutive_failures += 1
                            errors.append(f"Failed to fetch question {qid}")
                    except Exception as e:
                        chunk_failures += 1
                        consecutive_failures += 1
                        errors.append(f"Error fetching question {qid}: {str(e)}")

            all_questions.extend(chunk_results)

            # Update progress
            if progress_bar:
                progress = 0.15 + (len(all_questions) / total_questions) * 0.85
                progress_bar.progress(min(progress, 1.0))

            # Adaptive rate limiting: if we're seeing failures, slow down
            if chunk_failures > len(chunk) * 0.1:  # More than 10% failures
                rate_limited = True
                if status_text:
                    status_text.text(
                        f"Phase 2/2: Rate limited detected, pausing {RATE_LIMIT_BACKOFF}s..."
                    )
                time.sleep(RATE_LIMIT_BACKOFF)
            elif consecutive_failures >= 5:
                # Multiple consecutive failures - longer backoff
                rate_limited = True
                if status_text:
                    status_text.text(
                        f"Phase 2/2: Multiple failures, pausing {RATE_LIMIT_BACKOFF * 2}s..."
                    )
                time.sleep(RATE_LIMIT_BACKOFF * 2)
                consecutive_failures = 0
            elif chunk_idx < len(chunks) - 1:
                # Normal delay between chunks (not after the last one)
                time.sleep(CHUNK_DELAY)

        if progress_bar:
            progress_bar.progress(1.0)

        return {
            "questions": all_questions,
            "total_reported": total_results,
            "total_fetched": len(all_questions),
            "errors": errors,
            "cancelled": False,
        }


# =============================================================================
# Search Functions
# =============================================================================


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def create_searchable_text(question: Dict[str, Any]) -> str:
    """Create a combined searchable text from question and answer."""
    parts = []

    if question.get("questionText"):
        parts.append(clean_html(question["questionText"]))

    if question.get("answerText"):
        parts.append(clean_html(question["answerText"]))

    if question.get("heading"):
        parts.append(clean_html(question["heading"]))

    return " ".join(parts)


def create_query(query_string: str, ignore_case: bool = True) -> Query:
    """Create a Query object with appropriate parameters for the library version."""
    try:
        return Query(query_string, ignore_case=ignore_case, ignore_accent=True)
    except TypeError:
        try:
            return Query(query_string, ignore_case=ignore_case)
        except TypeError:
            return Query(query_string)


def create_search_query(query_string: str, ignore_case: bool = True):
    """Create a SearchQuery object for finding match positions."""
    if not SEARCH_QUERY_AVAILABLE:
        return None
    try:
        return SearchQuery(query_string, ignore_case=ignore_case)
    except Exception:
        return None


def highlight_matches(text: str, query_string: str, ignore_case: bool = True) -> str:
    """
    Highlight matching terms in text using SearchQuery.
    Returns HTML with highlighted spans.
    """
    if not text or not query_string.strip():
        return html.escape(text) if text else ""

    if not SEARCH_QUERY_AVAILABLE:
        # Fallback: simple highlighting without SearchQuery
        return highlight_matches_simple(text, query_string, ignore_case)

    try:
        search_query = create_search_query(query_string, ignore_case)
        if not search_query:
            return highlight_matches_simple(text, query_string, ignore_case)

        matches = search_query(text)

        if not matches:
            return html.escape(text)

        # Sort matches by start position (descending) to insert from end
        # Extract span from match objects
        match_spans = []
        for match in matches:
            if hasattr(match, "span"):
                # span is a tuple (start, end)
                if isinstance(match.span, tuple):
                    match_spans.append(match.span)
                else:
                    # Try to get span as attribute
                    try:
                        match_spans.append((match.span[0], match.span[1]))
                    except:
                        pass
            elif hasattr(match, "start") and hasattr(match, "end"):
                match_spans.append((match.start, match.end))

        if not match_spans:
            return html.escape(text)

        # Remove overlapping spans (keep longer ones)
        match_spans = sorted(match_spans, key=lambda x: (x[0], -(x[1] - x[0])))
        non_overlapping = []
        last_end = -1
        for start, end in match_spans:
            if start >= last_end:
                non_overlapping.append((start, end))
                last_end = end

        # Build highlighted text from end to start
        result = text
        for start, end in sorted(non_overlapping, reverse=True):
            matched_text = html.escape(result[start:end])
            result = (
                result[:start]
                + f'<span style="{HIGHLIGHT_STYLE}">{matched_text}</span>'
                + result[end:]
            )

        # Escape any remaining unescaped parts
        # Actually, we need to be more careful - escape parts that aren't in spans
        # Let's rebuild properly
        result_parts = []
        last_pos = 0
        for start, end in sorted(non_overlapping):
            # Add escaped text before this match
            result_parts.append(html.escape(text[last_pos:start]))
            # Add highlighted match
            matched_text = html.escape(text[start:end])
            result_parts.append(
                f'<span style="{HIGHLIGHT_STYLE}">{matched_text}</span>'
            )
            last_pos = end
        # Add remaining text
        result_parts.append(html.escape(text[last_pos:]))

        return "".join(result_parts)

    except Exception as e:
        # Fallback on any error
        return highlight_matches_simple(text, query_string, ignore_case)


def highlight_matches_simple(
    text: str, query_string: str, ignore_case: bool = True
) -> str:
    """
    Simple fallback highlighting - extracts quoted terms and highlights them.
    """
    if not text or not query_string.strip():
        return html.escape(text) if text else ""

    # Extract terms from query (words in quotes)
    terms = re.findall(r'"([^"]+)"', query_string)

    # Also get unquoted words that aren't operators
    operators = {"AND", "OR", "NOT", "and", "or", "not"}
    unquoted = re.sub(r'"[^"]+"', "", query_string)
    for word in unquoted.split():
        word = word.strip("()")
        if word and word not in operators:
            terms.append(word)

    if not terms:
        return html.escape(text)

    # Escape the text first
    escaped_text = html.escape(text)

    # Highlight each term
    for term in terms:
        if not term.strip():
            continue
        # Handle wildcards by converting to regex
        if "*" in term:
            pattern = re.escape(term).replace(r"\*", r"\w*")
        else:
            pattern = re.escape(term)

        flags = re.IGNORECASE if ignore_case else 0

        def replace_match(m):
            return f'<span style="{HIGHLIGHT_STYLE}">{m.group(0)}</span>'

        escaped_text = re.sub(pattern, replace_match, escaped_text, flags=flags)

    return escaped_text


def perform_boolean_search(
    questions: List[Dict[str, Any]], query_string: str, ignore_case: bool = True
) -> List[Dict[str, Any]]:
    """Perform boolean search on questions using eldar."""
    if not ELDAR_AVAILABLE:
        st.error("eldar/eldar_extended library not installed.")
        return []

    if not query_string.strip():
        return questions

    try:
        query = create_query(query_string, ignore_case=ignore_case)
    except Exception as e:
        st.error(f"Invalid query syntax: {str(e)}")
        return []

    matching = []
    for q in questions:
        searchable_text = create_searchable_text(q)
        try:
            if query(searchable_text):
                matching.append(q)
        except Exception:
            continue

    return matching


# =============================================================================
# Copy Format Functions (matching Chrome extension)
# =============================================================================


def format_name(name: str, house: str = "") -> str:
    """
    Remove honorific prefixes from names for Commons members only.
    Peer titles (Lord, Baroness, etc.) are preserved for Lords members.
    """
    if not name:
        return name

    # Only remove commoner honorifics (Mr, Mrs, Miss, Ms)
    # Keep peer titles and honours (Lord, Baroness, Sir, Dame, Dr, etc.)
    return re.sub(r"^(Mr|Mrs|Miss|Ms) ", "", name)


def abbreviate_party(party: str) -> str:
    """Convert party name to abbreviation."""
    if not party:
        return party
    return PARTY_ABBREVIATIONS.get(party, party)


def get_question_url(question: Dict[str, Any]) -> str:
    """Generate the Parliament website URL for a question."""
    uin = question.get("uin", "")
    date_tabled = question.get("dateTabled", "")

    if date_tabled:
        try:
            dt = datetime.fromisoformat(date_tabled.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = date_tabled[:10] if len(date_tabled) >= 10 else ""
    else:
        date_str = ""

    return f"https://questions-statements.parliament.uk/written-questions/detail/{date_str}/{uin}"


def format_question_text(text: str) -> str:
    """Process question text - remove 'To ask' prefix."""
    if not text:
        return ""
    cleaned = clean_html(text)
    if cleaned.startswith("To ask "):
        cleaned = cleaned[7:]
    elif cleaned.startswith("To ask"):
        cleaned = cleaned[6:].lstrip()
    return cleaned


def decapitalise_first_letter(text: str) -> str:
    """Lowercase the first letter of text."""
    if not text:
        return text
    return text[0].lower() + text[1:] if len(text) > 1 else text.lower()


def get_member_display_info(member: Optional[Dict], house: str) -> Tuple[str, str]:
    """
    Get formatted member name and party/constituency string.

    Returns:
        Tuple of (formatted_name_html, party_constituency_string)
    """
    if not member:
        return "<b>Unknown</b>", ""

    name = format_name(member.get("name", member.get("listAs", "Unknown")))
    party = abbreviate_party(member.get("party", ""))

    # Use partyAbbreviation if available and party wasn't abbreviated
    if not party or party == member.get("party", ""):
        party_abbr = member.get("partyAbbreviation", "")
        if party_abbr:
            party = party_abbr

    constituency = member.get("memberFrom", "")

    is_commons = house and house.strip() == "Commons"

    if is_commons:
        formatted_name = f"<b>{name} MP</b>"
        if party and constituency:
            party_const = f"({party}, {constituency})"
        elif party:
            party_const = f"({party})"
        else:
            party_const = ""
    else:
        formatted_name = f"<b>{name}</b>"
        party_const = f"({party})" if party else ""

    return formatted_name, party_const


def generate_copy_wq_html(question: Dict[str, Any]) -> str:
    """
    Generate HTML for 'Copy WQ' format.
    Format: <b>Name MP</b> (Party, Constituency) tabled a <a href="URL">Written Question</a> asking [question]
    """
    house = question.get("house", "")
    asking_member = question.get("askingMember")
    formatted_name, party_const = get_member_display_info(asking_member, house)

    url = get_question_url(question)
    linked_wq = f'<a href="{url}">Written Question</a>'

    question_text = format_question_text(question.get("questionText", ""))

    composed = (
        f"{formatted_name} {party_const} tabled a {linked_wq} asking {question_text}"
    )
    return f'<span style="font-size:10pt">{composed}</span>'


def generate_copy_answer_html(question: Dict[str, Any]) -> str:
    """
    Generate HTML for 'Copy Answer' format.
    Format: <b>Name MP</b> (Party, Constituency) replied that [answer]
    """
    house = question.get("house", "")
    answering_member = question.get("answeringMember")
    formatted_name, party_const = get_member_display_info(answering_member, house)

    answer_text = question.get("answerText", "")
    if answer_text:
        # Clean HTML but preserve some structure
        answer_text = clean_html(answer_text)
        answer_text = decapitalise_first_letter(answer_text)

    composed = f"{formatted_name} {party_const} replied that {answer_text}"
    return f'<span style="font-size:10pt">{composed}</span>'


def generate_copy_title_html(question: Dict[str, Any]) -> str:
    """
    Generate HTML for 'Copy Title' format.
    Format: <b>Name MP</b> (Party, Constituency) WQ on <a href="URL">Title</a>
    """
    house = question.get("house", "")
    asking_member = question.get("askingMember")
    formatted_name, party_const = get_member_display_info(asking_member, house)

    url = get_question_url(question)
    title = question.get("heading", "No heading")
    linked_title = f'<a href="{url}">{title}</a>'

    composed = f"{formatted_name} {party_const} WQ on {linked_title}"
    return f'<span style="font-size:10pt">{composed}</span>'


def create_copy_button_html(
    html_content: str, button_id: str, button_text: str, button_color: str = "#4A90D9"
) -> str:
    """
    Create an HTML/JS component that copies rich HTML to clipboard when clicked.
    """
    # Escape the HTML content for JavaScript string
    escaped_content = (
        html_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    )

    return f"""
    <button id="{button_id}" onclick="copyRichText_{button_id}()" style="
        background-color: {button_color};
        color: white;
        border: none;
        padding: 4px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        margin-right: 5px;
        transition: background-color 0.3s;
    ">{button_text}</button>
    <script>
    function copyRichText_{button_id}() {{
        const htmlContent = `{escaped_content}`;
        const blob = new Blob([htmlContent], {{ type: 'text/html' }});
        const clipboardItem = new ClipboardItem({{ 'text/html': blob }});
        navigator.clipboard.write([clipboardItem]).then(() => {{
            const btn = document.getElementById('{button_id}');
            const originalText = btn.innerText;
            const originalBg = btn.style.backgroundColor;
            btn.innerText = 'Copied! 🎉';
            btn.style.backgroundColor = '#00B74A';
            setTimeout(() => {{
                btn.innerText = originalText;
                btn.style.backgroundColor = originalBg;
            }}, 1300);
        }}).catch(err => {{
            const btn = document.getElementById('{button_id}');
            btn.innerText = 'Failed';
            btn.style.backgroundColor = '#f44336';
            setTimeout(() => {{
                btn.innerText = '{button_text}';
                btn.style.backgroundColor = '{button_color}';
            }}, 1300);
        }});
    }}
    </script>
    """


def create_copy_link_button_html(url: str, button_id: str) -> str:
    """
    Create an HTML/JS component that copies a plain text URL to clipboard when clicked.
    """
    return f"""
    <button id="{button_id}" onclick="copyLink_{button_id}()" style="
        background-color: #7F8C8D;
        color: white;
        border: none;
        padding: 4px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        margin-right: 5px;
        transition: background-color 0.3s;
    ">🔗 Copy Link</button>
    <script>
    function copyLink_{button_id}() {{
        navigator.clipboard.writeText("{url}").then(() => {{
            const btn = document.getElementById('{button_id}');
            const originalText = btn.innerText;
            const originalBg = btn.style.backgroundColor;
            btn.innerText = 'Copied! 🎉';
            btn.style.backgroundColor = '#00B74A';
            setTimeout(() => {{
                btn.innerText = originalText;
                btn.style.backgroundColor = originalBg;
            }}, 1300);
        }}).catch(err => {{
            const btn = document.getElementById('{button_id}');
            btn.innerText = 'Failed';
            btn.style.backgroundColor = '#f44336';
            setTimeout(() => {{
                btn.innerText = '🔗 Copy Link';
                btn.style.backgroundColor = '#7F8C8D';
            }}, 1300);
        }});
    }}
    </script>
    """


# =============================================================================
# Display Functions
# =============================================================================


def format_date(date_str: Optional[str]) -> str:
    """Format a date string for display."""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y")
    except (ValueError, AttributeError):
        return date_str


def get_member_name(member: Optional[Dict]) -> str:
    """Extract member name from member object."""
    if not member:
        return "Unknown"
    return member.get("name", member.get("listAs", "Unknown"))


def display_question(
    question: Dict[str, Any],
    search_query: str = "",
    ignore_case: bool = True,
    expanded: bool = False,
):
    """Display a single question in an expandable card with optional highlighting."""

    uin = question.get("uin", "N/A")
    heading = question.get("heading") or "No heading"
    date_tabled = format_date(question.get("dateTabled"))
    house = question.get("house", "Unknown")

    asking_member = get_member_name(question.get("askingMember"))
    answering_member = get_member_name(question.get("answeringMember"))
    answering_body = question.get("answeringBodyName", "Unknown")

    is_answered = question.get("dateAnswered") is not None
    date_answered = format_date(question.get("dateAnswered"))

    if question.get("isWithdrawn"):
        status = "🔴 Withdrawn"
    elif is_answered:
        status = "✅ Answered"
    else:
        status = "⏳ Awaiting"

    with st.expander(
        f"**{uin}** - {heading[:60]}{'...' if len(heading) > 60 else ''} [{status}]",
        expanded=expanded,
    ):
        # Show heading with highlighting if search query matches the heading
        if search_query.strip():
            highlighted_heading = highlight_matches(heading, search_query, ignore_case)
            # Only show the highlighted heading if it actually contains highlights
            if f'style="{HIGHLIGHT_STYLE}"' in highlighted_heading:
                st.markdown(f"**Topic:** {highlighted_heading}", unsafe_allow_html=True)
                st.markdown("")  # Small spacing

        # Copy buttons row
        st.markdown("**📋 Copy Formats:**")

        # Generate unique IDs for this question's buttons
        unique_id = str(uuid.uuid4()).replace("-", "")[:8]

        # Generate HTML content for each format
        wq_html = generate_copy_wq_html(question)
        title_html = generate_copy_title_html(question)
        url = get_question_url(question)

        # Build the combined buttons HTML - Title first, then Question
        buttons_html = f"""
        <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">
            {create_copy_button_html(title_html, f"title_{unique_id}", "📌 Copy Title", "#9B59B6")}
            {create_copy_button_html(wq_html, f"wq_{unique_id}", "📝 Copy Question", "#4A90D9")}
        """

        # Add Copy Answer button if answered
        if is_answered and question.get("answerText"):
            answer_html = generate_copy_answer_html(question)
            buttons_html += create_copy_button_html(
                answer_html, f"ans_{unique_id}", "💬 Copy Answer", "#27AE60"
            )

        # Add Copy Link button
        buttons_html += create_copy_link_button_html(url, f"link_{unique_id}")

        buttons_html += "</div>"

        # Render the buttons using components.html
        components.html(buttons_html, height=45)

        # Compact metadata section - all on fewer lines with reduced spacing
        meta_parts = [
            f"**UIN:** {uin}",
            f"**House:** {house}",
            f"**Tabled:** {date_tabled}",
            f"**By:** {asking_member}",
            f"**To:** {answering_body}",
        ]
        if is_answered:
            meta_parts.append(f"**Answered:** {date_answered} by {answering_member}")

        st.markdown(" · ".join(meta_parts), unsafe_allow_html=True)

        st.markdown("---")

        # Question text with highlighting
        st.markdown("**Question:**")
        question_text = clean_html(question.get("questionText", "No question text"))
        if search_query.strip():
            highlighted_question = highlight_matches(
                question_text, search_query, ignore_case
            )
            st.markdown(highlighted_question, unsafe_allow_html=True)
        else:
            st.markdown(question_text)

        # Answer text with highlighting
        if is_answered and question.get("answerText"):
            st.markdown("---")
            st.markdown("**Answer:**")
            answer_text = clean_html(question.get("answerText", ""))
            if search_query.strip():
                highlighted_answer = highlight_matches(
                    answer_text, search_query, ignore_case
                )
                st.markdown(highlighted_answer, unsafe_allow_html=True)
            else:
                st.markdown(answer_text)

        question_id = question.get("id")
        if question_id:
            st.markdown(f"[View on Parliament website]({get_question_url(question)})")


# =============================================================================
# Main Application
# =============================================================================


def main():
    st.set_page_config(
        page_title="Parliamentary Questions Search", page_icon="🏛️", layout="wide"
    )

    st.title("🏛️ UK Parliamentary Written Questions Search")

    # Initialise session state
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    if "filtered_questions" not in st.session_state:
        st.session_state["filtered_questions"] = []
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""
    if "ignore_case" not in st.session_state:
        st.session_state["ignore_case"] = True
    if "stop_flag" not in st.session_state:
        st.session_state["stop_flag"] = threading.Event()
    if "answering_body_filter" not in st.session_state:
        st.session_state["answering_body_filter"] = "All"
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    # Show library status
    if ELDAR_AVAILABLE:
        if USING_EXTENDED:
            st.caption(
                "✅ Using eldar_extended (wildcard support + highlighting enabled)"
            )
        else:
            st.caption("ℹ️ Using standard eldar library")
    else:
        st.error("⚠️ eldar library not found! Please install it.")
        return

    # ==========================================================================
    # Step 1: Load Questions (Sidebar)
    # ==========================================================================

    st.sidebar.header("1️⃣ Load Questions")

    # Answer status filter (moved up so we can use it to control date type)
    answered_filter = st.sidebar.selectbox(
        "Answer Status",
        options=["Answered", "Unanswered", "All"],
        index=0,  # Default to "Answered"
        help="Filter by whether questions have been answered",
    )

    # Date type selector - only show "Date Answered" option if filtering by Answered
    st.sidebar.subheader("Date Filter")

    if answered_filter == "Answered":
        date_type_options = ["Date Answered", "Date Tabled"]
        date_type_index = 0  # Default to Date Answered for answered questions
    else:
        date_type_options = ["Date Tabled"]
        date_type_index = 0

    date_type_display = st.sidebar.selectbox(
        "Filter by",
        options=date_type_options,
        index=date_type_index,
        help="Choose whether to filter by when questions were tabled or answered",
    )

    # Map display value to API value
    date_type = "answered" if date_type_display == "Date Answered" else "tabled"

    # Date range
    st.sidebar.caption(f"Maximum range: {MAX_DATE_RANGE_DAYS} days")

    col1, col2 = st.sidebar.columns(2)

    default_from = datetime.now() - timedelta(days=1)
    default_to = datetime.now()

    with col1:
        date_from = st.date_input(
            "From", value=default_from, max_value=datetime.now(), format="DD/MM/YYYY"
        )

    with col2:
        date_to = st.date_input(
            "To", value=default_to, max_value=datetime.now(), format="DD/MM/YYYY"
        )

    # Validate date range
    date_range_days = (date_to - date_from).days
    if date_range_days > MAX_DATE_RANGE_DAYS:
        st.sidebar.error(
            f"Date range cannot exceed {MAX_DATE_RANGE_DAYS} days. Currently: {date_range_days} days"
        )
        date_range_valid = False
    elif date_range_days < 0:
        st.sidebar.error("'From' date must be before 'To' date")
        date_range_valid = False
    else:
        st.sidebar.success(f"Range: {date_range_days + 1} day(s)")
        date_range_valid = True

    # House selection
    house = st.sidebar.selectbox("House", options=["Both", "Commons", "Lords"], index=0)

    # Map answered filter to API values
    answered_api_value = {
        "All": None,
        "Answered": "Answered",
        "Unanswered": "Unanswered",
    }.get(answered_filter)

    # Load button
    st.sidebar.markdown("---")
    load_button = st.sidebar.button(
        "📥 Load Questions",
        type="primary",
        use_container_width=True,
        disabled=not date_range_valid,
    )

    # Show loaded data info
    if st.session_state["questions"]:
        st.sidebar.markdown("---")

        # Verification indicator
        if "load_stats" in st.session_state:
            stats = st.session_state["load_stats"]
            fetched = stats["total_fetched"]
            reported = stats["total_reported"]

            if fetched == reported:
                st.sidebar.success(f"✅ **Loaded:** {fetched} / {reported} questions")
            elif fetched < reported:
                st.sidebar.warning(f"⚠️ **Loaded:** {fetched} / {reported} questions")
                if stats.get("errors"):
                    st.sidebar.caption(
                        f"Some requests failed ({len(stats['errors'])} errors)"
                    )
                elif stats.get("cancelled"):
                    st.sidebar.caption("Loading was cancelled")

            if stats.get("errors"):
                with st.sidebar.expander("View errors"):
                    for err in stats["errors"][:5]:  # Show max 5 errors
                        st.caption(err)

        if "load_params" in st.session_state:
            params = st.session_state["load_params"]
            st.sidebar.caption(f"Date type: {params['date_type']}")
            st.sidebar.caption(f"From: {params['date_from']}")
            st.sidebar.caption(f"To: {params['date_to']}")
            st.sidebar.caption(f"House: {params['house']}")
            st.sidebar.caption(f"Status: {params['answered']}")

        if st.sidebar.button("🗑️ Clear & Start New Search", use_container_width=True):
            st.session_state["questions"] = []
            st.session_state["filtered_questions"] = []
            st.session_state["search_query"] = ""
            st.session_state.pop("load_params", None)
            st.session_state.pop("load_stats", None)
            st.rerun()

    # Handle load button
    if load_button and date_range_valid:
        date_from_dt = datetime.combine(date_from, datetime.min.time())
        date_to_dt = datetime.combine(date_to, datetime.max.time())

        api = ParliamentaryQuestionsAPI()

        # Reset stop flag
        st.session_state["stop_flag"] = threading.Event()

        # Create placeholders for progress and cancel button
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        cancel_placeholder = st.sidebar.empty()

        # Show cancel button
        if cancel_placeholder.button("⏹️ Cancel Loading", use_container_width=True):
            st.session_state["stop_flag"].set()

        status_text.text("Starting fetch...")

        result = api.fetch_all_questions(
            date_from_dt,
            date_to_dt,
            date_type=date_type,
            house=house if house != "Both" else None,
            answered=answered_api_value,
            max_results=MAX_RESULTS,
            progress_bar=progress_bar,
            status_text=status_text,
            stop_flag=st.session_state["stop_flag"],
        )

        # Clear progress elements
        progress_bar.empty()
        status_text.empty()
        cancel_placeholder.empty()

        if result["cancelled"]:
            st.sidebar.warning(
                f"Loading cancelled. Fetched {result['total_fetched']} questions before stopping."
            )

        if result["questions"]:
            st.session_state["questions"] = result["questions"]
            st.session_state["filtered_questions"] = result["questions"]
            st.session_state["search_query"] = ""
            st.session_state["load_params"] = {
                "date_type": date_type_display,
                "date_from": date_from.strftime("%d %b %Y"),
                "date_to": date_to.strftime("%d %b %Y"),
                "house": house,
                "answered": answered_filter,
            }
            st.session_state["load_stats"] = {
                "total_reported": result["total_reported"],
                "total_fetched": result["total_fetched"],
                "errors": result["errors"],
                "cancelled": result["cancelled"],
            }
            st.rerun()
        elif not result["cancelled"]:
            st.sidebar.warning("No questions found for the selected filters.")

    # ==========================================================================
    # Step 2: Main Content - Two Column Layout
    # ==========================================================================

    if not st.session_state["questions"]:
        st.info("👈 Use the sidebar to load questions, then search through them here.")

        with st.expander("📖 Query Syntax Help"):
            st.markdown("""
            ### Boolean Operators
            - **AND**: Both terms must be present → `"climate" AND "energy"`
            - **OR**: Either term can be present → `"NHS" OR "healthcare"`
            - **NOT / AND NOT**: Exclude terms → `"budget" AND NOT "defence"`
            - **Parentheses**: Group terms → `("NHS" OR "healthcare") AND "funding"`

            ### Examples
            - Simple term: `"immigration"`
            - Multiple terms: `"artificial intelligence" OR "AI"`
            - Complex query: `("climate change" OR "global warming") AND "policy" AND NOT "denial"`
            - Wildcard (eldar-extended only): `"climat*"` matches "climate", "climatic", etc.

            ### Tips
            - Wrap phrases in double quotes: `"written question"`
            - The search looks in both the question text and the answer text
            - Case is ignored by default
            - Matching terms are highlighted in yellow in results
            """)
        return

    # Two column layout
    left_col, right_col = st.columns([1, 2])

    # ==========================================================================
    # Left Column: Search Controls
    # ==========================================================================

    with left_col:
        st.header("2️⃣ Search")

        # Get unique answering bodies from loaded questions for filter
        answering_bodies = sorted(
            set(
                q.get("answeringBodyName", "Unknown")
                for q in st.session_state.get("questions", [])
                if q.get("answeringBodyName")
            )
        )

        # Use a form to handle the search properly
        with st.form(key="search_form"):
            query = st.text_area(
                "Boolean search query:",
                value=st.session_state.get("search_query", ""),
                placeholder='e.g., ("NHS" OR "healthcare") AND "funding"',
                help="Enter a boolean query using AND, OR, NOT operators",
                height=100,
            )

            ignore_case = st.checkbox("Ignore case", value=True)

            # Answering body filter
            if answering_bodies:
                selected_body = st.selectbox(
                    "Filter by Answering Body (optional):",
                    options=["All"] + answering_bodies,
                    index=0,
                    help="Limit results to questions answered by a specific department",
                )
            else:
                selected_body = "All"

            col1, col2 = st.columns(2)
            with col1:
                search_button = st.form_submit_button(
                    "🔍 Search", type="primary", use_container_width=True
                )
            with col2:
                show_all_button = st.form_submit_button(
                    "📋 Show All", use_container_width=True
                )

        # Handle search
        if search_button:
            st.session_state["search_query"] = query
            st.session_state["ignore_case"] = ignore_case
            st.session_state["answering_body_filter"] = selected_body

            # Start with all questions
            questions_to_search = st.session_state["questions"]

            # Apply answering body filter first
            if selected_body and selected_body != "All":
                questions_to_search = [
                    q
                    for q in questions_to_search
                    if q.get("answeringBodyName") == selected_body
                ]

            # Then apply boolean search
            if query.strip():
                filtered = perform_boolean_search(
                    questions_to_search, query, ignore_case=ignore_case
                )
                st.session_state["filtered_questions"] = filtered
            else:
                st.session_state["filtered_questions"] = questions_to_search

            # Reset to page 1 when search changes
            st.session_state["current_page"] = 1
            st.rerun()

        if show_all_button:
            st.session_state["search_query"] = ""
            st.session_state["answering_body_filter"] = "All"
            st.session_state["filtered_questions"] = st.session_state["questions"]
            st.session_state["current_page"] = 1
            st.rerun()

        # Query help
        with st.expander("📖 Query Syntax Help"):
            st.markdown(
                """
            **Operators:**
            - `AND` - Both terms required
            - `OR` - Either term matches
            - `NOT` / `AND NOT` - Exclude term
            - `()` - Group terms
            - `""` - Exact phrase
            - `*` - Wildcard (extended only)

            **Examples:**
            ```
            "immigration"
            "NHS" OR "healthcare"
            ("climate" OR "environment") AND "policy"
            "fund*" AND NOT "defence"
            ```

            **Note:** Matching terms are <span style="background-color: #FFEB3B;">highlighted</span> in results.
            """,
                unsafe_allow_html=True,
            )

        # Results summary
        st.markdown("---")
        filtered = st.session_state.get("filtered_questions", [])
        total = len(st.session_state.get("questions", []))

        st.metric("Results", f"{len(filtered)} / {total}")

        if st.session_state.get("search_query"):
            st.caption(f"Query: `{st.session_state['search_query']}`")

        if (
            st.session_state.get("answering_body_filter")
            and st.session_state.get("answering_body_filter") != "All"
        ):
            st.caption(f"Filtered by: {st.session_state['answering_body_filter']}")

    # ==========================================================================
    # Right Column: Results
    # ==========================================================================

    with right_col:
        st.header("Results")

        filtered = st.session_state.get("filtered_questions", [])
        current_query = st.session_state.get("search_query", "")
        ignore_case = st.session_state.get("ignore_case", True)

        if filtered:
            # Pagination setup
            items_per_page = 15
            total_pages = (len(filtered) - 1) // items_per_page + 1

            # Initialise page in session state if not present
            if "current_page" not in st.session_state:
                st.session_state["current_page"] = 1

            # Ensure page is within valid range
            if st.session_state["current_page"] > total_pages:
                st.session_state["current_page"] = total_pages
            if st.session_state["current_page"] < 1:
                st.session_state["current_page"] = 1

            page = st.session_state["current_page"]

            # Top controls row: Pagination and Expand All
            col_page, col_expand = st.columns([3, 1])

            with col_page:
                if total_pages > 1:
                    col_prev, col_select, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button(
                            "◀ Prev",
                            key="prev_top",
                            disabled=(page <= 1),
                            use_container_width=True,
                        ):
                            st.session_state["current_page"] = page - 1
                            st.rerun()
                    with col_select:
                        new_page = st.selectbox(
                            "Page",
                            options=range(1, total_pages + 1),
                            index=page - 1,
                            format_func=lambda x: f"Page {x} of {total_pages}",
                            key="results_page_top",
                            label_visibility="collapsed",
                        )
                        if new_page != page:
                            st.session_state["current_page"] = new_page
                            st.rerun()
                    with col_next:
                        if st.button(
                            "Next ▶",
                            key="next_top",
                            disabled=(page >= total_pages),
                            use_container_width=True,
                        ):
                            st.session_state["current_page"] = page + 1
                            st.rerun()

            with col_expand:
                expand_all = st.checkbox("Expand All", value=False, key="expand_all")

            # Ensure page is still valid
            if page > total_pages:
                page = total_pages
                st.session_state["current_page"] = page
            if page < 1:
                page = 1
                st.session_state["current_page"] = page

            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            # Display questions with highlighting
            for question in filtered[start_idx:end_idx]:
                display_question(
                    question, current_query, ignore_case, expanded=expand_all
                )

            # Bottom navigation (only show if more than one page)
            if total_pages > 1:
                st.markdown("---")
                col_prev_b, col_info, col_next_b = st.columns([1, 2, 1])
                with col_prev_b:
                    if st.button(
                        "◀ Previous Page",
                        key="prev_bottom",
                        disabled=(page <= 1),
                        use_container_width=True,
                    ):
                        st.session_state["current_page"] = page - 1
                        st.rerun()
                with col_info:
                    st.markdown(
                        f"<p style='text-align: center; padding-top: 8px;'>Page {page} of {total_pages}</p>",
                        unsafe_allow_html=True,
                    )
                with col_next_b:
                    if st.button(
                        "Next Page ▶",
                        key="next_bottom",
                        disabled=(page >= total_pages),
                        use_container_width=True,
                    ):
                        st.session_state["current_page"] = page + 1
                        st.rerun()
        else:
            st.info(
                "No questions match your search criteria. Try adjusting your query or click 'Show All'."
            )

    # Footer
    st.markdown("---")
    st.caption(
        "Data sourced from the [UK Parliament Written Questions API](https://questions-statements-api.parliament.uk/)"
    )


if __name__ == "__main__":
    main()
