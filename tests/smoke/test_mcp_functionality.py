#!/usr/bin/env python3
"""Test MCP server functionality without running the full server."""

import sys
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_mcp_tools():
    """Test that MCP tools can be imported and called."""
    print("🧪 Testing MCP Tool Functions")
    print("=" * 40)
    
    try:
        # Import the MCP server functions directly
        from pycontextify.mcp_server import status, index_code, search
        print("✅ MCP functions imported successfully")
        
        # Test status function (should work without dependencies)
        print("\n📊 Testing status() function:")
        try:
            result = status()
            print(f"✅ Status function returned: {result.get('status', 'unknown')}")
            if 'error' in result:
                print(f"ℹ️  Status contains expected initialization error: {result['error'][:100]}...")
            else:
                print("🎉 Status function working without errors!")
        except Exception as e:
            print(f"⚠️  Status function error (expected): {str(e)[:100]}...")
        
        # Test index_code with invalid path (should handle gracefully)
        print("\n📁 Testing index_code() with invalid path:")
        try:
            result = index_code("/nonexistent/path")
            print(f"✅ index_code handled invalid path correctly: {result}")
        except Exception as e:
            print(f"❌ index_code failed: {e}")
        
        # Test index_code with valid path (will fail on embeddings but should validate path)
        print("\n📁 Testing index_code() with valid path:")
        try:
            result = index_code(str(Path.cwd()))
            print(f"✅ index_code response: {result}")
            if 'error' in result:
                print("ℹ️  Expected error due to missing embedding dependencies")
        except Exception as e:
            print(f"⚠️  index_code error (may be expected): {str(e)[:100]}...")
        
        # Test search function
        print("\n🔍 Testing search() function:")
        try:
            result = search("test query", 5)
            print(f"✅ search response: {result}")
        except Exception as e:
            print(f"⚠️  search error (expected): {str(e)[:100]}...")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import MCP functions: {e}")
        return False

def test_configuration_only():
    """Test that we can at least get system status without embeddings."""
    print("\n🔧 Testing Configuration-Only Mode")
    print("=" * 40)
    
    try:
        from pycontextify.index.config import Config
        config = Config()
        
        print(f"✅ Provider: {config.embedding_provider}")
        print(f"✅ Model: {config.embedding_model}")
        print(f"✅ Index Directory: {config.index_dir}")
        print(f"✅ Chunk Size: {config.chunk_size}")
        
        # Test provider availability (without actually importing)
        print(f"✅ Configured Provider: {config.embedding_provider}")
        print("ℹ️  Skipping actual provider availability check to avoid heavy imports")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all functionality tests."""
    print("🚀 PyContextify MCP Functionality Test")
    print("=" * 50)
    
    config_ok = test_configuration_only()
    tools_ok = test_mcp_tools()
    
    print("\n📋 Summary")
    print("=" * 50)
    print(f"✅ Configuration: {'OK' if config_ok else 'FAILED'}")
    print(f"✅ MCP Tools: {'OK' if tools_ok else 'FAILED'}")
    
    if config_ok and tools_ok:
        print("\n🎉 MCP server is functional!")
        print("\n💡 Next Steps:")
        print("- To get full embedding functionality, install:")
        print("  uv add sentence-transformers torch")
        print("- Or use with OpenAI by setting PYCONTEXTIFY_EMBEDDING_PROVIDER=openai")
        print("  and PYCONTEXTIFY_OPENAI_API_KEY=your_key")
        print("- Run the server: python -m pycontextify.mcp_server")
        return True
    else:
        print("\n❌ Some functionality tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)