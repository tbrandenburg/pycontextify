#!/usr/bin/env python3
"""Test HuggingFace connectivity and model loading."""

import socket
import ssl
import time
import requests
from transformers import AutoTokenizer


def test_dns_resolution():
    """Test DNS resolution for HuggingFace."""
    print("Testing DNS resolution...")
    start = time.time()
    try:
        ip = socket.gethostbyname('huggingface.co')
        duration = time.time() - start
        print(f"✅ DNS lookup successful: {ip} (took {duration:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ DNS lookup failed: {e}")
        return False


def test_ssl_connection():
    """Test SSL connection to HuggingFace."""
    print("Testing SSL connection...")
    start = time.time()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection(('huggingface.co', 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname='huggingface.co') as ssock:
                pass
        duration = time.time() - start
        print(f"✅ SSL connection successful (took {duration:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ SSL connection failed: {e}")
        return False


def test_api_request():
    """Test HuggingFace API request."""
    print("Testing HuggingFace API request...")
    start = time.time()
    try:
        response = requests.get('https://huggingface.co/api/models', timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            print(f"✅ API request successful (took {duration:.2f}s)")
            return True
        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return False


def test_model_config_download():
    """Test downloading model config."""
    print("Testing model config download...")
    start = time.time()
    try:
        url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
        response = requests.get(url, timeout=30)
        duration = time.time() - start
        if response.status_code == 200:
            print(f"✅ Model config download successful (took {duration:.2f}s)")
            return True
        else:
            print(f"❌ Model config download failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model config download failed: {e}")
        return False


def test_tokenizer_loading():
    """Test loading a HuggingFace tokenizer."""
    print("Testing tokenizer loading (this might take a while on first run)...")
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        duration = time.time() - start
        print(f"✅ Tokenizer loading successful (took {duration:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False


def main():
    """Run all connectivity tests."""
    print("🔍 Testing HuggingFace connectivity...\n")
    
    tests = [
        ("DNS Resolution", test_dns_resolution),
        ("SSL Connection", test_ssl_connection), 
        ("API Request", test_api_request),
        ("Model Config Download", test_model_config_download),
        ("Tokenizer Loading", test_tokenizer_loading),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:25} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All connectivity tests passed! Your connection to HuggingFace is working.")
    else:
        print("⚠️  Some tests failed. Check the details above for specific issues.")


if __name__ == "__main__":
    main()