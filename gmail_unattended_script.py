#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised Gmail Script to:
1. Authenticate with Gmail's API.
2. Retrieve all unread emails from the past 24 hours (not just 10).
3. Use OpenAI to classify and summarize the emails.
4. Mark non-important emails as 'READ'.
5. Generate and beautify an executive-level HTML summary.
6. Send the summary to your own email address.

Usage:
    python gmail_unattended_script.py

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
    pip install openai
"""

import os
import sys
import logging
import pickle
import base64
import json
import openai
import datetime

from email.mime.text import MIMEText
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------------------------------------------------------
# Globals and Configuration
# ------------------------------------------------------------------------------

# Update scopes if you need different Gmail permissions:
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Adjust paths as needed:
CREDENTIALS_PATH = "secrets/credentials.json"
TOKEN_PATH = "secrets/token.pickle"

# Get OpenAI API key from `secrets/openai.json`:
openai.api_key = json.loads(open("secrets/openai.json").read())["api_key"]

# We will use the 'gpt-4' or 'gpt-3.5-turbo' or your chosen model for classification
OPENAI_MODEL_CLASSIFY = "gpt-4o"

# We will use some sample 'gpt-4' or 'gpt-3.5-turbo' for the final summary
OPENAI_MODEL_SUMMARY = "o1-preview"

# Replace with your own "send-to" email address:
SEND_TO_EMAIL = "blgottshall@gmail.com"


# ------------------------------------------------------------------------------
# 1. Gmail API Authentication
# ------------------------------------------------------------------------------
def get_gmail_service() -> object:
    """
    Authenticates (or refreshes) with the Gmail API using OAuth.
    Returns an authenticated service object.
    """
    logging.info("Authenticating with Gmail API...")
    creds = None

    # 1) Check if token already exists:
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token_file:
            creds = pickle.load(token_file)

    # 2) If no valid creds or they are expired, do the OAuth flow:
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            logging.info("Initiating OAuth flow for new credentials...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # 3) Save the credentials for future use:
        with open(TOKEN_PATH, "wb") as token_file:
            pickle.dump(creds, token_file)

    # 4) Build the Gmail service:
    service = build("gmail", "v1", credentials=creds)
    logging.info("Gmail service created successfully.")
    return service


# ------------------------------------------------------------------------------
# 2. Retrieve All Unread Emails from the Past 24 Hours
# ------------------------------------------------------------------------------
def get_unread_emails_past_24h(service):
    """
    Fetches *all* unread emails from the last 24 hours (removes any maxResults limit).

    :param service: Authenticated Gmail service instance.
    :return: A list of email data dicts with "id", "subject", "snippet".
    """
    logging.info("Fetching all unread emails from the past 24 hours...")

    query = "is:unread newer_than:1d"
    emails_data = []
    page_token = None

    try:
        while True:
            response = (
                service.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    pageToken=page_token,
                )
                .execute()
            )
            messages = response.get("messages", [])
            logging.info(f"Found {len(messages)} messages in this page.")

            for msg in messages:
                msg_detail = (
                    service.users().messages().get(userId="me", id=msg["id"]).execute()
                )
                snippet = msg_detail.get("snippet", "")
                headers = msg_detail.get("payload", {}).get("headers", [])

                # Attempt to get the subject:
                subject = None
                for h in headers:
                    if h["name"].lower() == "subject":
                        subject = h["value"]
                        break

                emails_data.append(
                    {
                        "id": msg["id"],
                        "subject": subject or "(No Subject)",
                        "snippet": snippet,
                    }
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                # No more pages
                break

        logging.info(f"Total unread emails from past 24 hrs: {len(emails_data)}")
        return emails_data

    except Exception as e:
        logging.error(f"Error while fetching emails: {e}")
        return []


# ------------------------------------------------------------------------------
# 3. OpenAI Classification & Summarization
# ------------------------------------------------------------------------------
CLASSIFY_AND_SUMMARIZE_FUNCTION = {
    "name": "classify_and_summarize",
    "description": (
        "Classify an email into exactly one category: "
        "newsletter, notification, or important. Also provide "
        "a short, neatly formatted Markdown summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Which category the email belongs to. Must be exactly one of:\n\n"
                    "1) newsletter\n"
                    "2) notification\n"
                    "3) important"
                ),
                "enum": ["newsletter", "notification", "important"],
            },
            "summary": {
                "type": "string",
                "description": (
                    "A concise Markdown summary of the key points from the email. "
                    "Include relevant next steps or instructions if the email references "
                    "new invitations, messages, or any call to action. Label these steps "
                    "appropriately. Keep it neatly formatted in MD."
                ),
            },
        },
        "required": ["category", "summary"],
        "additionalProperties": False,
    },
}


def classify_and_summarize_email(subject: str, snippet: str) -> dict:
    """
    Uses OpenAI's function calling to classify the email
    as either 'newsletter', 'notification', or 'important',
    and generate a short Markdown summary.

    :param subject: The email's subject line.
    :param snippet: The email's snippet text.
    :return: Dictionary with fields "category" and "summary".
    """
    system_instructions = (
        "You are an email classification and summarization service. "
        "Categorize each email into exactly one of: 'newsletter', 'notification', or 'important'. "
        "Then produce a concise summary in Markdown, carefully extracting key points without "
        "omitting vital information. If the email references new invitations, messages, or calls to action, "
        "include them. Only use details from the subject/snippet. Do not invent or guess details."
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": f"Subject: {subject}\n\nEmail snippet:\n{snippet}"},
    ]

    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL_CLASSIFY,
            messages=messages,
            functions=[CLASSIFY_AND_SUMMARIZE_FUNCTION],
            function_call={"name": "classify_and_summarize"},
        )
        choice = response.choices[0]
        # Replace .get("function_call") with direct attribute:
        function_call_obj = choice.message.function_call
        if not function_call_obj:
            logging.warning("No function_call returned; defaulting to 'important'.")
            return {
                "category": "important",
                "summary": "Unable to parse email with function calling. Defaulting to 'important'.",
            }

        # Access arguments directly:
        args_str = function_call_obj.arguments
        args_dict = json.loads(args_str)
        category = args_dict.get("category", "important")
        summary_md = args_dict.get("summary", "")

        logging.debug(f"[DEBUG] Email classified as {category}. Summary: {summary_md}")
        return {"category": category, "summary": summary_md}

    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return {
            "category": "important",
            "summary": (
                f"Error occurred. Defaulting to category 'important'.\n\n**Details:** {str(e)}"
            ),
        }


# ------------------------------------------------------------------------------
# 4. Mark Non-Important Emails as Read
# ------------------------------------------------------------------------------
def mark_non_important_as_read(service, classified_emails):
    """
    Marks all emails in 'classified_emails' that are NOT 'important' as 'READ'.
    """
    logging.info("Marking non-important emails as read...")
    for email in classified_emails:
        category = email["category"]
        message_id = email["id"]
        if category != "important":
            try:
                service.users().messages().modify(
                    userId="me",
                    id=message_id,
                    body={"removeLabelIds": ["UNREAD"]},
                ).execute()
                logging.info(
                    f"Marked message {message_id} as read (category: {category})."
                )
            except Exception as e:
                logging.error(f"Could not mark message {message_id} as read: {e}")


# ------------------------------------------------------------------------------
# 5. Generate Executive-Level Recap
# ------------------------------------------------------------------------------
def get_current_date_string() -> str:
    """
    Return today's date in a nice format.
    """
    return datetime.datetime.now().strftime("%B %d, %Y")


def generate_executive_summary(classified_emails) -> str:
    """
    Generates an executive-level briefing from the given list of classified emails.
    Relies on content already summarized in Markdown.
    Combines them into a single Markdown doc that highlights urgent or critical matters first.
    """

    # We'll do a naive grouping by category or keywords:
    important_emails = [e for e in classified_emails if e["category"] == "important"]
    notification_emails = [
        e for e in classified_emails if e["category"] == "notification"
    ]
    newsletter_emails = [e for e in classified_emails if e["category"] == "newsletter"]

    lines = []
    lines.append(f"**Executive Summary for {get_current_date_string()}**\n")

    # Important:
    if important_emails:
        lines.append("### Important Emails\n")
        for i, email in enumerate(important_emails, start=1):
            lines.append(f"**{i}.** **Subject:** {email['subject']}")
            lines.append(email["summary"])
            lines.append("")
    # Notification:
    if notification_emails:
        lines.append("### Notifications\n")
        for i, email in enumerate(notification_emails, start=1):
            lines.append(f"**{i}.** **Subject:** {email['subject']}")
            lines.append(email["summary"])
            lines.append("")
    # Newsletter:
    if newsletter_emails:
        lines.append("### Newsletters\n")
        for i, email in enumerate(newsletter_emails, start=1):
            lines.append(f"**{i}.** **Subject:** {email['subject']}")
            lines.append(email["summary"])
            lines.append("")

    lines.append("\n**End of Summary**")
    combined_md = "\n".join(lines)
    logging.info("Executive summary (Markdown) generated.")
    return combined_md


# ------------------------------------------------------------------------------
# 6. Convert Markdown to (simple) HTML
# ------------------------------------------------------------------------------
def convert_md_to_html(markdown_text: str) -> str:
    """
    Sends the markdown text to OpenAI and asks for a simple HTML rendering.
    """
    prompt = f"Convert the following Markdown to HTML.\n\n{markdown_text}"
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL_SUMMARY,
            messages=[{"role": "user", "content": prompt}],
        )
        html_output = response.choices[0].message.content.strip()
        # Remove the first and last line as they are code fences:
        html_output = "\n".join(html_output.split("\n")[1:-1])
        logging.info("Markdown converted to HTML successfully.")
        return html_output
    except Exception as e:
        logging.error(f"Error converting Markdown to HTML: {e}")
        return f"<p>Error converting Markdown to HTML: {e}</p>"


# ------------------------------------------------------------------------------
# 7. Send Summary via Email
# ------------------------------------------------------------------------------
def send_html_email(service, to_address, subject, html_content):
    """
    Sends an HTML email via the provided Gmail service.
    """
    logging.info(f"Sending final summary email to {to_address}...")
    message = MIMEText(html_content, "html")
    message["to"] = to_address
    message["subject"] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    body = {"raw": raw_message}

    try:
        sent_msg = service.users().messages().send(userId="me", body=body).execute()
        logging.info(f"Email sent successfully! Message ID: {sent_msg['id']}")
    except Exception as e:
        logging.error(f"An error occurred while sending email: {e}")


# ------------------------------------------------------------------------------
# Main Script Execution
# ------------------------------------------------------------------------------
def main():
    # 1. Get Gmail service
    service = get_gmail_service()

    # 2. Fetch all unread from last 24 hrs
    unread_emails = get_unread_emails_past_24h(service)

    if not unread_emails:
        logging.info("No unread emails found in the past 24 hours.")
        return

    # 3. Classify and Summarize
    logging.info("Classifying and summarizing emails with OpenAI...")
    classified_emails = []
    for i, email in enumerate(unread_emails, start=1):
        subject = email["subject"]
        snippet = email["snippet"]
        logging.info(f"Processing Email #{i}: Subject='{subject}'")

        result = classify_and_summarize_email(subject, snippet)
        email["category"] = result["category"]
        email["summary"] = result["summary"]
        classified_emails.append(email)

    # 4. Mark non-important as read
    mark_non_important_as_read(service, classified_emails)

    # 5. Generate an executive-level summary
    summary_md = generate_executive_summary(classified_emails)

    # 6. Convert that summary to HTML
    summary_html = convert_md_to_html(summary_md)

    # 7. Send the final summary back to your inbox
    send_html_email(
        service=service,
        to_address=SEND_TO_EMAIL,
        subject=f"Daily Recap: {get_current_date_string()}",
        html_content=summary_html,
    )

    logging.info("Script completed.")


if __name__ == "__main__":
    main()
