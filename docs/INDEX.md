# Documentation Index

Complete documentation guide for PyContextify MCP server.

## Documentation Structure

### Core Guides

| Document | Description | Audience |
|----------|-------------|----------|
| **[MCP_SERVER.md](MCP_SERVER.md)** | Complete MCP server guide | Users & Developers |
| **[TESTING.md](TESTING.md)** | Testing guide and strategies | Developers |
| **[BOOTSTRAP.md](BOOTSTRAP.md)** | Index bootstrapping | Advanced Users |

---

## Quick Links

### For Users

**Getting Started:**
1. [Start the MCP Server](MCP_SERVER.md#quick-start)
2. [Available Tools](MCP_SERVER.md#mcp-tools)
3. [Claude Desktop Integration](MCP_SERVER.md#integration-with-claude-desktop)

**Common Tasks:**
- [Index documents](MCP_SERVER.md#tool-examples)
- [Index codebase](MCP_SERVER.md#tool-examples)
- [Configure the server](MCP_SERVER.md#command-line-options)

### For Developers

**Development:**
1. [Run tests](TESTING.md#quick-start)
2. [Test organization](TESTING.md#test-organization)
3. [Coverage analysis](TESTING.md#coverage-analysis)

**Contributing:**
- [Test writing guidelines](TESTING.md#best-practices)
- [CI/CD recommendations](TESTING.md#cicd-recommendations)

---

## Document Summaries

### MCP_SERVER.md
**Complete guide to running and using the PyContextify MCP server**

Topics covered:
- Server startup and configuration
- 5 MCP tools (status, index_document, index_code, search, reset_index)
- Command-line options and environment variables
- Claude Desktop integration
- Usage examples
- Troubleshooting

**Start here if you're:** Setting up PyContextify for the first time

### TESTING.md
**Complete guide to testing PyContextify**

Topics covered:
- Test organization (unit, integration, system)
- Test execution strategies
- Performance optimization (78% faster with fast mode)
- Coverage analysis (71% overall)
- Complete user flow validation
- CI/CD recommendations

**Start here if you're:** Developing or contributing to PyContextify

### BOOTSTRAP.md
**Index bootstrapping and distribution**

Topics covered:
- Pre-built index creation
- Index distribution via HTTP/file URLs
- Bootstrap configuration
- Use cases

**Start here if you're:** Distributing pre-built indexes

---

## Documentation Coverage

### What's Documented

✅ **MCP Server**
- Complete command-line reference
- All 5 MCP tools with examples
- Configuration options
- Integration guides

✅ **Testing**
- All test types (unit, integration, system)
- Execution strategies
- Performance benchmarks
- Coverage reports

✅ **Features**
- Index bootstrapping
- Multi-source indexing

✅ **Troubleshooting**
- Common issues
- Debug commands
- Performance tuning

### What's Missing

❌ Contributor guide (coming soon)  
❌ API reference (coming soon)  
❌ Architecture deep dive (coming soon)

---

## Quick Reference

### Essential Commands

**Start server:**
```bash
uv run pycontextify
```

**Run tests:**
```bash
uv run pytest tests/ -m "not slow"
```

**Check coverage:**
```bash
uv run pytest tests/ --cov=pycontextify
```

### Key Statistics

| Metric | Value |
|--------|-------|
| **MCP Tools** | 5 |
| **Test Cases** | 285 |
| **Code Coverage** | 71% |
| **Fast Test Time** | ~21s |
| **Full Test Time** | ~98s |
| **Supported Content** | Documents, Code |

---

## Getting Help

### Documentation
1. Check the relevant guide above
2. Search for your topic in the specific document
3. Look for examples and troubleshooting sections

### Testing
- See [TESTING.md](TESTING.md) for complete test documentation
- Run `pytest tests/ -v` to see all available tests
- Run `pytest --markers` to see test markers

### Issues
- Check [troubleshooting sections](MCP_SERVER.md#troubleshooting)
- Run diagnostic commands
- Check test output for hints

---

## Document Maintenance

### Updating Documentation

When adding features:
1. Update relevant guide (MCP_SERVER.md, TESTING.md, etc.)
2. Add examples if applicable
3. Update this README if structure changes

When fixing issues:
1. Add to troubleshooting section
2. Include diagnostic commands
3. Document workarounds

### Documentation Standards

- Use Markdown formatting
- Include code examples
- Provide command output samples
- Keep guides focused and concise
- Link between related documents

---

## Version

**Documentation Version**: 1.0  
**Last Updated**: 2025-01-08  
**PyContextify Version**: 0.1.0

---

## Project Links

- **Main README**: [../README.md](../README.md)
- **GitHub**: https://github.com/pycontextify/pycontextify
- **Issues**: https://github.com/pycontextify/pycontextify/issues
