"""
Keep Streamlit app awake by pinging its URL.
Run manually or on a schedule (Task Scheduler, cron).
"""
import sys
import urllib.request

URL = "https://hurricanepaths.streamlit.app/"

try:
    with urllib.request.urlopen(URL, timeout=30) as response:
        status = response.getcode()
        print(f"Pinged {URL} -> {status}")
        sys.exit(0 if status == 200 else 1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
