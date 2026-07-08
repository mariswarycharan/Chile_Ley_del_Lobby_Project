# test_gsheet_auth.py
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Point this directly at the freshly downloaded JSON file
creds = Credentials.from_service_account_file(
    r"credentials\service_account.json",
    scopes=SCOPES
)

client = gspread.authorize(creds)

sheet_id = "1jQ-vdL8HQuWGgAfyhVpOspGpR9Hh_YCbc-9jUY6q7BQ"
spreadsheet = client.open_by_key(sheet_id)

print("✅ SUCCESS! Connected to:", spreadsheet.title)
print("Worksheets:", [ws.title for ws in spreadsheet.worksheets()])