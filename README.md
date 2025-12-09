# UK Parliamentary Written Questions Search

A Streamlit application for searching UK Parliamentary Written Questions using boolean search queries.

## Features

- **Date Range Selection**: Filter questions by tabled date
- **House Filter**: Search Commons, Lords, or both
- **Boolean Search**: Powerful query syntax with AND, OR, NOT operators
- **Search Both Question & Answer**: Searches through question text and ministerial responses
- **CSV Export**: Download results for further analysis
- **Rate Limiting**: Respectful API usage to avoid overwhelming the Parliament servers

## Installation on macOS

### Prerequisites

1. **Python 3.9+** - Check your version:
   ```bash
   python3 --version
   ```
   
   If you don't have Python or need to upgrade:
   ```bash
   # Using Homebrew
   brew install python@3.11
   ```

### Step-by-Step Setup

1. **Create a project directory**:
   ```bash
   mkdir ~/pq_search
   cd ~/pq_search
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Copy the application files** into the directory:
   - `app.py` - Main application
   - `requirements.txt` - Dependencies

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**: Streamlit will automatically open `http://localhost:8501` in your browser.

### Installing eldar-extended (Optional)

The basic `eldar` package provides boolean search. For wildcard support and additional features, you can install the extended version:

```bash
# Clone the eldar-extended repository (assuming it's available)
git clone https://github.com/your-repo/eldar-extended.git
cd eldar-extended
pip install -e .
```

## Usage Guide

### Basic Search

1. **Set Date Range**: Use the sidebar to select start and end dates
2. **Select House**: Choose Commons, Lords, or Both
3. **Enter Query**: Type your boolean search query
4. **Click Search**: Fetch and filter questions

### Boolean Query Syntax

| Operator | Description | Example |
|----------|-------------|---------|
| `AND` | Both terms required | `"NHS" AND "funding"` |
| `OR` | Either term matches | `"NHS" OR "healthcare"` |
| `NOT` / `AND NOT` | Exclude term | `"budget" AND NOT "defence"` |
| `()` | Group terms | `("NHS" OR "healthcare") AND "crisis"` |
| `""` | Exact phrase | `"climate change"` |
| `*` | Wildcard (extended only) | `"climat*"` |

### Example Queries

```
# Simple keyword
"immigration"

# Phrase search
"artificial intelligence"

# OR query
"NHS" OR "National Health Service"

# Complex query
("climate change" OR "global warming") AND "policy" AND NOT "denial"

# Department-specific
"Home Office" AND ("visa" OR "asylum")

# Wildcard (requires eldar-extended)
"environ*" AND "regulat*"
```

### Search Options

- **Ignore Case**: Matches regardless of capitalisation (default: on)
- **Exact Word Match**: Requires exact word boundaries (default: off)
  - When ON: "movie" won't match "movies"
  - When OFF: "movie" will match "movies"

## API Rate Limiting

The application includes built-in rate limiting to be respectful of the Parliament API:

- **100 requests per minute** maximum
- **0.5 second delay** between paginated requests
- **Configurable maximum results** (default: 1000, max: 5000)

### Adjusting Rate Limits

If you need to modify rate limits, edit these constants in `app.py`:

```python
RATE_LIMIT_REQUESTS = 100  # Max requests per window
RATE_LIMIT_WINDOW = 60     # Window in seconds
REQUEST_DELAY = 0.5        # Delay between requests
```

**Note**: The Parliament API doesn't officially document rate limits, but being conservative helps ensure reliable access.

## Troubleshooting

### Common Issues

**1. "eldar library not found"**
```bash
pip install eldar
```

**2. "Connection timeout"**
- Check your internet connection
- The Parliament API may be temporarily unavailable
- Try again later or reduce the date range

**3. "Too many results"**
- Narrow your date range
- Reduce the "Maximum results" slider
- Add more specific search terms

**4. Slow performance**
- Large date ranges require many API calls
- Each page fetches 100 questions
- Use a narrower date range for faster results

### Virtual Environment Issues

If you're having dependency conflicts:

```bash
# Deactivate current environment
deactivate

# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Fields

The application retrieves the following fields for each question:

| Field | Description |
|-------|-------------|
| UIN | Unique Identification Number |
| Heading | Subject heading |
| House | Commons or Lords |
| Date Tabled | When the question was submitted |
| Asked By | MP/Peer who asked |
| Answering Body | Government department |
| Date Answered | When answered (if applicable) |
| Answered By | Minister who responded |
| Question Text | Full question text |
| Answer Text | Full answer text (if answered) |
| Is Withdrawn | Whether question was withdrawn |

## Architecture

```
┌─────────────────┐
│   Streamlit UI  │
├─────────────────┤
│  Boolean Search │  ← eldar/eldar-extended
├─────────────────┤
│   API Client    │  ← Rate limiting
├─────────────────┤
│ Parliament API  │
└─────────────────┘
```

## API Documentation

The UK Parliament Written Questions API is publicly available:
- **Base URL**: `https://questions-statements-api.parliament.uk`
- **Documentation**: Available as OpenAPI specification
- **No authentication required**

## Contributing

Feel free to modify the application for your needs. Key files:

- `app.py`: Main application logic
- `requirements.txt`: Python dependencies

## Licence

This application is provided as-is for searching publicly available parliamentary data.

## Useful Links

- [UK Parliament Written Questions](https://questions-statements.parliament.uk/written-questions)
- [Parliament API](https://questions-statements-api.parliament.uk)
- [eldar Boolean Search](https://github.com/kerighan/eldar)
