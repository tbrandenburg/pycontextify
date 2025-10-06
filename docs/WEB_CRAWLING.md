# Recursive Web Crawling

## Overview

This document describes the recursive web crawling functionality in PyContextify, including identified issues, implemented improvements, and usage guidance. The improvements are inspired by best practices from Scrapy, a leading web scraping framework.

---

## Original Issues Identified

### Issue 1: Hard-Coded Link Limit
**Problem:** Only the first 10 links per page were followed (hard-coded in source)
- When a page had 30+ links, only 10 would be crawled
- Not configurable by users
- Too restrictive for comprehensive documentation indexing

**Impact:** Incomplete indexing of documentation sites, particularly navigation/menu pages with many links.

### Issue 2: Non-Intuitive Depth Calculation  
**Problem:** The `max_depth` parameter was off-by-one
- `max_depth=2` only crawled root + direct children (2 levels, not 3)
- Due to starting depth at 0 and condition `current_depth < max_depth - 1`
- Confusing for users expecting inclusive depth counting

**Impact:** Users had to set `max_depth=3` to crawl what they thought would be depth 2.

### Issue 3: Lack of Transparency
**Problem:** No logging when crawl limits were reached
- Users didn't know when link limits truncated their crawl
- No visibility into depth limit behavior

**Impact:** Silent failures led to incomplete indexing without user awareness.

---

## Improvements Implemented

### 1. Configurable Link Limits

**Change:** Added `max_links_per_page` parameter
- **Default:** 50 links per page (suitable for comprehensive documentation)
- **Range:** 1-100 links
- **Rationale:** This MCP is designed for trustworthy indexing of documentation. Users expect comprehensive coverage, not artificially limited crawling.

**Why 50?** Strikes a balance between:
- Comprehensive enough for most documentation sites
- Not so high as to cause excessive crawling
- Still respects rate limiting via `delay_seconds`
- Users can easily adjust up to 100 or down as needed

### 2. Fixed Depth Calculation

**Change:** Depth now works inclusively and intuitively
- Depth starts at 1 instead of 0
- Condition changed from `current_depth < max_depth - 1` to `max_depth == 0 or current_depth < max_depth`
- `max_depth=1`: only the starting URL
- `max_depth=2`: starting URL + direct children  
- `max_depth=3`: starting URL + children + grandchildren
- `max_depth=0`: unlimited depth (use with caution)

### 3. Enhanced Logging

**Change:** Added transparent logging at multiple levels:
- **INFO level:** Number of links found on each page, current crawl depth
- **WARNING level:** When link limit is reached and some links are skipped
- **DEBUG level:** When depth limits are reached, already-visited URLs

**Benefit:** Users can now understand and optimize their crawl behavior.

---

## Implementation Details

### Depth Calculation

The new depth calculation follows these rules:

```python
# Check depth limit (inclusive)
if max_depth > 0 and current_depth > max_depth:
    return []  # Stop crawling

# Extract and follow links
if max_depth == 0 or current_depth < max_depth:
    # Continue crawling to the next level
```

### Link Limiting

Links are limited per page with transparent logging:

```python
links_to_crawl = links[:self.max_links_per_page]

if total_links > self.max_links_per_page:
    logger.warning(
        f"Found {total_links} links on {url}, "
        f"but limiting to first {self.max_links_per_page} "
        f"(set max_links_per_page to increase)"
    )
```

## Usage Examples

### Basic Usage (Default Settings)

```python
from pycontextify.index.loaders import WebpageLoader

# Default: max_depth=2, max_links_per_page=50
loader = WebpageLoader()
pages = loader.load(
    "https://example.com",
    recursive=True,
    max_depth=2
)
```

### Custom Link Limit

```python
# Increase link limit for very comprehensive crawl
loader = WebpageLoader(max_links_per_page=100)
pages = loader.load(
    "https://example.com",
    recursive=True,
    max_depth=2
)
```

### Unlimited Depth (Use with Caution)

```python
# Crawl unlimited depth
# WARNING: Can result in very large crawls!
loader = WebpageLoader(max_links_per_page=50)
pages = loader.load(
    "https://example.com",
    recursive=True,
    max_depth=0  # 0 = unlimited
)
```

### Via MCP Server

When using the MCP server, the new parameters are exposed:

```python
# MCP tool call
index_webpage(
    url="https://example.com",
    recursive=True,
    max_depth=3,           # Inclusive: root + 2 more levels
    max_links_per_page=20  # Follow up to 20 links per page
)
```

## Best Practices (From Scrapy)

### Depth Limits
- **Start Small:** Begin with `max_depth=2` and increase as needed
- **Typical Range:** Most crawls use depths between 2-5
- **Unlimited Depth:** Use `max_depth=0` only when absolutely necessary

### Link Limits
- **Default (50):** Suitable for comprehensive documentation indexing
- **Higher Values (75-100):** For very large documentation sites with many cross-references
- **Lower Values (10-25):** For targeted crawls or rate-sensitive sites
- **Very Low (5):** For minimal sampling or testing

### Logging
- Monitor logs to understand crawl behavior
- `WARNING` messages indicate when limits are hit
- Adjust limits based on log feedback

---

## Testing

**New Test Suite:** `tests/test_recursive_crawling.py`
- ✅ 5 comprehensive tests
- ✅ Tests depth calculation, link limits, and configurable parameters
- ✅ Documents actual vs expected behavior
- ✅ All tests passing

```bash
# Run recursive crawling tests
uv run pytest tests/test_recursive_crawling.py -v -s

# Run full test suite with coverage
uv run pytest tests/ --cov=pycontextify --cov-report=term-missing
```

**Test Results:**
- 260 tests passed
- 68% code coverage
- All recursive crawling scenarios validated

---

## Troubleshooting Common Issues

### Issue: Only 1 page indexed in recursive crawl

**Possible Causes:**
1. **Domain mismatch:** Only same-domain links are followed
2. **Depth too low:** Try increasing `max_depth`
3. **Link limit hit:** Increase `max_links_per_page`
4. **No links found:** Check page structure

**Solution:**
```python
# Increase both depth and link limits
index_webpage(
    url="https://docs.example.com",
    recursive=True,
    max_depth=3,
    max_links_per_page=100
)
```

### Issue: Crawling taking too long

**Solutions:**
- Reduce `max_depth` (try depth=2)
- Reduce `max_links_per_page` (try 25 or 10)  
- Adjust `delay_seconds` (but respect rate limits!)
- Consider indexing specific pages individually

---

## Migration from Old Behavior

### Depth Value Adjustments

If you were using the old behavior:
- Old `max_depth=2` → New `max_depth=2` (same value, but now more intuitive)
- Old `max_depth=3` → New `max_depth=3` (same value, but now clearer)

The numeric values don't need to change, but the behavior is now clearer:
- New: `max_depth=2` means "2 levels deep" (root=1, children=2)
- Old: `max_depth=2` was confusing (root=0, children=1)

### Link Limit Changes

**Before:** Hard-coded 10 links per page  
**After:** Default 50 links per page (configurable 1-100)

If you need the old behavior: set `max_links_per_page=10`

---

## References

- **Scrapy Documentation:** https://docs.scrapy.org/en/latest/topics/settings.html#depth-limit
- **Scrapy DEPTH_LIMIT Setting:** Inclusive depth limit for crawling
- **Web Crawling Best Practices:** Balance between breadth, depth, and politeness

## Conclusion

These improvements bring PyContextify's web crawling capabilities in line with industry-standard practices, making it:
- More intuitive to use
- More transparent in operation
- More configurable for different use cases

The changes maintain backward compatibility while offering users better control and understanding of the crawling process.
