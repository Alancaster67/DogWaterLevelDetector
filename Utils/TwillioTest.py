import os
from dotenv import load_dotenv
from twilio.rest import Client

#loading Twillio secrets
load_dotenv()
account_sid = os.getenv("ACCOUNT_SID")
auth_token = os.getenv("AUTH_TOKEN")

client = Client(account_sid, auth_token)
message = client.messages.create(
    body = "No dog bowl detected under PiCam. Check dog bowl",
    from_ = "+18665184103",
    to = "+15206684049"
    )