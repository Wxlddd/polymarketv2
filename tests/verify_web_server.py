import asyncio
import aiohttp
import sys
from config.settings import SystemConfig
from src.ui.web_server import WebServer

class DummyOrchestrator:
    pass

async def main():
    print("=== Verification of Web Server & Bloomberg UI Endpoint ===")
    config = SystemConfig()
    
    # Override settings for verification (WebServerConfig is frozen, so write through __dict__)
    config.web_server.__dict__["HOST"] = "127.0.0.1"
    config.web_server.__dict__["PORT"] = 8888
    config.web_server.__dict__["ENABLED"] = True
    
    dummy_orch = DummyOrchestrator()
    server = WebServer(config, dummy_orch)
    
    print("1. Starting Web Server...")
    await server.start()
    print("[OK] Web Server started on http://127.0.0.1:8888")
    
    async with aiohttp.ClientSession() as session:
        # Test HTTP Root Page
        print("\n2. Requesting HTTP Root GET '/'...")
        async with session.get("http://127.0.0.1:8888/") as resp:
            print(f"  - Status Code: {resp.status} (Expected: 200)")
            content = await resp.text()
            print(f"  - HTML length: {len(content)} bytes")
            if resp.status == 200 and "<html" in content.lower():
                print("  - [OK] Page successfully fetched and contains HTML content.")
            else:
                print("  - [FAIL] Mismatched HTML response.")
                await server.stop()
                sys.exit(1)
                
        # Test Data Range Discovery Endpoint
        print("\n3. Requesting GET '/api/data_range'...")
        async with session.get("http://127.0.0.1:8888/api/data_range") as resp:
            print(f"  - Status Code: {resp.status} (Expected: 200)")
            res_json = await resp.json()
            print(f"  - Data Range response: {res_json}")
            if resp.status == 200 and res_json.get("success") and "min_iso" in res_json:
                print("  - [OK] Data range endpoint operates successfully.")
            else:
                print("  - [FAIL] Endpoint failed.")
                await server.stop()
                sys.exit(1)
                
        # Test WebSocket Handshake
        print("\n4. Performing WebSocket handshake '/ws/dashboard'...")
        async with session.ws_connect("ws://127.0.0.1:8888/ws/dashboard") as ws:
            print("  - [OK] WebSocket connection opened successfully.")
            
            # Send dynamic state mock payload
            print("  - Sending mock state payload to confirm client distribution...")
            mock_state = {"type": "portfolio_state", "ticker": "BTC", "equity": 10500.0, "cash": 9500.0}
            server.update_state(mock_state)
            
            # Wait for throttled broadcast
            await asyncio.sleep(0.5)
            
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                print(f"  - WebSocket received data: {data}")
                if data.get("ticker") == "BTC" and data.get("equity") == 10500.0:
                    print("  - [OK] Data delivered correctly over throttled channel.")
                else:
                    print("  - [FAIL] Mismatched payload data.")
                    await server.stop()
                    sys.exit(1)
            else:
                print(f"  - [FAIL] Received invalid WS message type: {msg.type}")
                await server.stop()
                sys.exit(1)
                
    print("\n5. Halting Web Server...")
    await server.stop()
    print("[OK] Web Server stopped cleanly.")
    print("\n=== All Web server unit checks PASSED ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
